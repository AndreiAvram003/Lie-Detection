import os
import numpy as np
import pandas as pd
import librosa
import joblib
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.src.utils import pad_sequences
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils.class_weight import compute_class_weight



# Importă modelul și funcția de extragere a caracteristicilor
# Să presupunem că extract_frame_based_features este definit în voice_processor
# sau într-un fișier utilitar. Pentru acest exemplu, o vom re-defini aici.
# from voice_processor import extract_frame_based_features
from ernn_model import build_ernn_model

# --- CONSTANTE DE CONFIGURARE (Ajustează-le în funcție de datele tale!) ---
SR_TARGET = 22050  # Rata de eșantionare țintă
N_MFCC = 13
FRAME_LENGTH_MS = 25
HOP_LENGTH_MS = 10

# MAX_SEQ_LENGTH: Determină această valoare după ce ai extras toate secvențele din antrenare
# (ex: percentila 95 a lungimilor sau o valoare fixă rezonabilă).
# Pentru acest exemplu, folosim o valoare placeholder.
# Aceasta va fi actualizată după analiza datelor.
MAX_SEQ_LENGTH_PLACEHOLDER = 250  # EXEMPLU: aprox 2.5 secunde cu hop_length_ms=10

# NUM_FEATURES_PER_FRAME: Calculează pe baza funcției de extracție
# 13(MFCC)+1(RMS)+1(ZCR)+1(Centroid)+1(Bandwidth)+7(Contrast)+1(Rolloff)+1(Pitch)+1(Tempo) = 27
NUM_FEATURES_PER_FRAME = 27

DATA_CSV_PATH = r'C:\Users\Andrei\Desktop\Faculta\Lie Detection\ai_model\voice_analysis\audio_features.csv'  # Conține 'file' (numele fișierului wav) și 'label'
AUDIO_CLIPS_DIR = r'C:\Users\Andrei\Desktop\Faculta\Lie Detection\ai_model\voice_analysis\audio_clips'  # Directorul cu fișierele .wav
SAVE_DIR = r'C:\Users\Andrei\Desktop\Faculta\Lie Detection\ai_model\voice_analysis'  # Director pentru salvarea modelului și scaler-ului


# --- SFÂRȘIT CONSTANTE ---

def extract_frame_based_features(audio_path: str, sr_target: int = SR_TARGET, n_mfcc: int = N_MFCC,
                                 frame_length_ms: int = FRAME_LENGTH_MS, hop_length_ms: int = HOP_LENGTH_MS):
    """
    Extrage caracteristici audio pe bază de ferestre temporale (frame-uri).
    Identică cu cea din voice_processor.py pentru consistență.
    """
    try:
        y, sr_orig = librosa.load(audio_path, sr=None)
        if sr_orig != sr_target:
            y = librosa.resample(y, orig_sr=sr_orig, target_sr=sr_target)
        sr = sr_target

        n_fft = int(sr * frame_length_ms / 1000)
        hop_length = int(sr * hop_length_ms / 1000)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfccs_transposed = mfccs.T

        if mfccs_transposed.shape[0] == 0: return None
        num_frames = mfccs_transposed.shape[0]
        all_frame_features = [mfccs_transposed]

        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        all_frame_features.append(rms[:num_frames].reshape(-1, 1))
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        all_frame_features.append(zcr[:num_frames].reshape(-1, 1))
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        all_frame_features.append(spectral_centroid[:num_frames].reshape(-1, 1))
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        all_frame_features.append(spectral_bandwidth[:num_frames].reshape(-1, 1))
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        all_frame_features.append(spectral_contrast.T[:num_frames, :])
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        all_frame_features.append(spectral_rolloff[:num_frames].reshape(-1, 1))

        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), frame_length=n_fft,
                                hop_length=hop_length)
        pitch_mean = np.nanmean(f0) if np.any(np.isfinite(f0)) else 0.0
        all_frame_features.append(np.full((num_frames, 1), pitch_mean))

        tempo_values, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        tempo = np.median(librosa.feature.tempo(y=y, sr=sr, hop_length=hop_length)) if len(tempo_values) > 0 else 60.0
        all_frame_features.append(np.full((num_frames, 1), tempo))

        final_sequence = np.concatenate(all_frame_features, axis=1)
        return np.nan_to_num(final_sequence, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        print(f"Eroare la extragerea caracteristicilor din {audio_path}: {e}")
        return None


def load_and_process_data(csv_path: str, audio_dir: str, max_seq_len_override: int = None):
    """
    Încarcă datele, extrage caracteristici secvențiale, antrenează scaler-ul și aplică padding.
    """
    df = pd.read_csv(csv_path)

    all_sequences = []
    labels = []
    sequence_lengths = []

    print("Extragerea caracteristicilor secvențiale din fișierele audio...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Presupunem că 'file' conține numele fișierului .wav (ex: 'lie_clip_1.wav')
        # Și că există o coloană 'label' numerică (0 sau 1) sau o deducem.
        audio_filename = row['file']
        label = row.get('label',
                        1 if 'lie' in audio_filename.lower() else 0)  # Dedu eticheta dacă nu există coloana 'label'

        audio_file_path = os.path.join(audio_dir, audio_filename)
        if not os.path.exists(audio_file_path):
            print(f"Avertisment: Fișierul audio {audio_file_path} nu a fost găsit. Se sare peste.")
            continue

        features_sequence = extract_frame_based_features(audio_file_path)
        if features_sequence is not None and features_sequence.shape[0] > 0:
            all_sequences.append(features_sequence)
            labels.append(label)
            sequence_lengths.append(features_sequence.shape[0])
        else:
            print(f"Avertisment: Nu s-au putut extrage caracteristici pentru {audio_filename}. Se sare peste.")

    if not all_sequences:
        raise ValueError("Nu s-au putut procesa fișierele audio. Verifică căile și fișierele.")

    # Determinarea MAX_SEQ_LENGTH dacă nu este specificată
    if max_seq_len_override:
        max_len = max_seq_len_override
        print(f"Se folosește MAX_SEQ_LENGTH specificat: {max_len}")
    else:
        max_len = int(np.percentile(sequence_lengths, 95))  # Sau altă valoare (mediană, medie, etc.)
        print(
            f"MAX_SEQ_LENGTH determinat (percentila 95): {max_len}. Lungime max. observată: {np.max(sequence_lengths)}")

    # Antrenarea Scaler-ului
    # Se concatenează toate frame-urile din toate secvențele pentru a antrena scaler-ul
    print("Antrenarea scaler-ului...")
    concatenated_frames = np.vstack([seq for seq in all_sequences if seq.shape[0] > 0])
    scaler = StandardScaler()
    scaler.fit(concatenated_frames)
    scaler_path = os.path.join(SAVE_DIR, "voice_frame_based_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler antrenat și salvat la: {scaler_path}")

    # Scalarea și Padding-ul secvențelor
    print("Scalarea și padding-ul secvențelor...")
    processed_X = []
    for seq in tqdm(all_sequences):
        if seq.shape[0] == 0: continue
        scaled_seq = scaler.transform(seq)
        padded_seq = \
        pad_sequences([scaled_seq], maxlen=max_len, dtype='float32', padding='post', truncating='post', value=0.0)[0]
        processed_X.append(padded_seq)

    X = np.array(processed_X)
    y = np.array(labels)

    return X, y, scaler, max_len


if __name__ == '__main__':
    # 1. Încărcare și Preprocesare Date
    # Folosim MAX_SEQ_LENGTH_PLACEHOLDER sau None pentru a lăsa scriptul să determine
    X_data, y_data, fitted_scaler, actual_max_seq_len = load_and_process_data(DATA_CSV_PATH, AUDIO_CLIPS_DIR,
                                                                              max_seq_len_override=MAX_SEQ_LENGTH_PLACEHOLDER)

    if X_data.shape[0] == 0:
        print("Nu s-au încărcat date. Verifică procesarea.")
        exit()

    print(f"Forma finală a datelor X: {X_data.shape}")  # (num_samples, actual_max_seq_len, NUM_FEATURES_PER_FRAME)
    print(f"Forma finală a etichetelor y: {y_data.shape}")

    # 2. Împărțire în Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )

    # 3. Construire Model
    # Input shape pentru model este (timesteps, features_per_timestep)
    model_input_shape = (actual_max_seq_len, X_data.shape[2])  # X_data.shape[2] este num_features_per_frame
    model = build_ernn_model(model_input_shape)
    model.summary()

    # 4. Callbacks și Ponderi Clase
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True,
                                   verbose=1)  # Patience mărită
    model_checkpoint = ModelCheckpoint(
        os.path.join(SAVE_DIR, "best_voice_frame_based_model.keras"),
        monitor='val_loss', save_best_only=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001,
                                  verbose=1)  # Factor și patience ajustate

    callbacks_list = [early_stopping, model_checkpoint, reduce_lr]

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Ponderi clase: {class_weight_dict}")

    # 5. Antrenare Model
    print("Pornirea antrenării modelului...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,  # Număr mai mare de epoci, EarlyStopping va opri optim
        batch_size=16,  # Batch size ajustat
        class_weight=class_weight_dict,
        callbacks=callbacks_list,
        verbose=1
    )

    # 6. Salvare Model Final (cel mai bun este deja salvat de ModelCheckpoint)
    # model.save(os.path.join(SAVE_DIR, "final_voice_frame_based_model.keras"))

    # 7. Evaluare și Vizualizare
    print("\nEvaluare pe setul de testare:")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Pierdere Test: {loss:.4f}")
    print(f"Acuratețe Test: {accuracy:.4f}")

    # Grafice
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_curves_voice_frame_based.png"))
    plt.show()

    # Matrice de Confuzie și Raport de Clasificare
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Voice Model (Frame-based)")
    plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix_voice_frame_based.png"))
    plt.show()

    print("\nRaport de Clasificare:")
    print(classification_report(y_test, y_pred, target_names=['Adevăr (0)', 'Minciună (1)']))

    print(f"\nIMPORTANT: MAX_SEQ_LENGTH folosit pentru antrenare a fost: {actual_max_seq_len}")
    print(f"Asigură-te că această valoare (sau una compatibilă) este folosită în 'voice_processor.py'.")