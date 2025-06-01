import os
import numpy as np
import librosa
import joblib
import tempfile

from keras.src.saving import load_model
from keras.src.utils import pad_sequences
from moviepy import VideoFileClip

# Import necesar

# --- CONSTANTE DE CONFIGURARE (Trebuie să fie IDENTICE cu cele din antrenare!) ---
SR_TARGET = 22050
N_MFCC = 13
FRAME_LENGTH_MS = 25
HOP_LENGTH_MS = 10

# !!! IMPORTANT !!!
# Această valoare TREBUIE să fie aceeași cu `actual_max_seq_len` determinată sau folosită
# în scriptul `train_voice_model.py` după procesarea datelor de antrenare.
# Va trebui să o actualizezi manual aici după ce rulezi scriptul de antrenare o dată
# și vezi ce valoare s-a folosit/determinat pentru MAX_SEQ_LENGTH.
# Sau salveaz-o într-un fișier de configurare.
# De exemplu, dacă antrenarea a folosit 250:
PREDICTION_MAX_SEQ_LENGTH = 250  # <<-- ACTUALIZEAZĂ ACEASTĂ VALOARE!

MODEL_DIR = r'C:\Users\Andrei\Desktop\Faculta\Lie Detection\ai_model\voice_analysis'


# --- SFÂRȘIT CONSTANTE ---

def extract_frame_based_features(audio_path: str, sr_target: int = SR_TARGET, n_mfcc: int = N_MFCC,
                                 frame_length_ms: int = FRAME_LENGTH_MS, hop_length_ms: int = HOP_LENGTH_MS):
    """
    Extrage caracteristici audio pe bază de ferestre temporale (frame-uri).
    Identică cu cea din train_voice_model.py pentru consistență.
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

        if mfccs_transposed.shape[0] == 0:
            print(f"Avertisment: Nu s-au putut extrage frame-uri MFCC din {audio_path}")
            return None
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
        print(f"Eroare critică la extragerea caracteristicilor din {audio_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_lie_from_audio(video_path: str,
                           model_name: str = "best_voice_frame_based_model.keras",
                           scaler_name: str = "voice_frame_based_scaler.pkl"):
    """
    Procesează un videoclip, extrage caracteristicile audio secvențiale (frame-based),
    scalează, aplică padding și face o predicție.
    """
    model_path = os.path.join(MODEL_DIR, model_name)
    scaler_path = os.path.join(MODEL_DIR, scaler_name)

    # Verifică existența PREDICTION_MAX_SEQ_LENGTH
    if PREDICTION_MAX_SEQ_LENGTH is None or PREDICTION_MAX_SEQ_LENGTH <= 0:
        print("[Eroare CRITICĂ] Constanta PREDICTION_MAX_SEQ_LENGTH nu este setată corect în voice_processor.py!")
        print("Aceasta trebuie actualizată cu valoarea folosită la antrenare.")
        return -1, 0.0

    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp_audio_file:
        try:
            video_clip = VideoFileClip(video_path)
            if video_clip.audio is None:
                print(f"[Eroare] Nu există componentă audio în fișierul: {video_path}")
                return -1, 0.0
            video_clip.audio.write_audiofile(tmp_audio_file.name, fps=SR_TARGET,
                                             logger=None)  # Specifică fps pentru consistență sr
            current_audio_path = tmp_audio_file.name
        except Exception as e:
            print(f"Eroare la extragerea componentei audio din video: {e}")
            return -1, 0.0

        features_sequence = extract_frame_based_features(current_audio_path)
        if features_sequence is None or features_sequence.shape[0] == 0:
            print(f"Nu s-au putut extrage caracteristici audio secvențiale pentru {video_path}")
            return -1, 0.0

        try:
            scaler = joblib.load(scaler_path)
        except FileNotFoundError:
            print(f"[Eroare CRITICĂ] Fișierul scaler '{scaler_path}' nu a fost găsit!")
            print("Asigură-te că ai antrenat și salvat scaler-ul corect (ex: voice_frame_based_scaler.pkl).")
            return -1, 0.0
        except Exception as e:
            print(f"Eroare la încărcarea scaler-ului: {e}")
            return -1, 0.0

        X_scaled = scaler.transform(features_sequence)

        # Padding/Trunchiere la PREDICTION_MAX_SEQ_LENGTH
        X_padded = pad_sequences([X_scaled], maxlen=PREDICTION_MAX_SEQ_LENGTH, dtype='float32',
                                 padding='post', truncating='post', value=0.0)[0]

        # Reshape pentru LSTM: (batch_size, num_timesteps, num_features_per_frame)
        # Modelul se așteaptă la un batch, chiar dacă e de 1.
        X_reshaped = X_padded.reshape((1, X_padded.shape[0], X_padded.shape[1]))

        try:
            model = load_model(model_path)
        except Exception as e:
            print(f"[Eroare CRITICĂ] Nu s-a putut încărca modelul de la '{model_path}': {e}")
            return -1, 0.0

        preds = model.predict(X_reshaped)
        final_prediction_probability = preds[0][0]

        predicted_class = 1 if final_prediction_probability > 0.5 else 0
        confidence = float(final_prediction_probability) if predicted_class == 1 else float(
            1 - final_prediction_probability)

    return predicted_class, confidence


if __name__ == '__main__':
    test_video = "calea/catre/videoclipul_tau_de_test.mp4"  # <<< MODIFICĂ ACEASTĂ CALE!

    if not os.path.exists(MODEL_DIR):
        print(f"Directorul specificat pentru model '{MODEL_DIR}' nu există. Verifică calea.")
    elif PREDICTION_MAX_SEQ_LENGTH is None or PREDICTION_MAX_SEQ_LENGTH <= 0:
        print(
            f"PREDICTION_MAX_SEQ_LENGTH nu este setată corect în {__file__}. Trebuie actualizată cu valoarea din antrenare.")
    elif not os.path.exists(test_video):
        print(f"Videoclipul de test '{test_video}' nu există. Verifică calea.")
    else:
        print(f"Se procesează videoclipul: {test_video}")
        print(f"Se folosește MAX_SEQ_LENGTH pentru predicție: {PREDICTION_MAX_SEQ_LENGTH}")
        print(f"Model așteptat: {os.path.join(MODEL_DIR, 'best_voice_frame_based_model.keras')}")
        print(f"Scaler așteptat: {os.path.join(MODEL_DIR, 'voice_frame_based_scaler.pkl')}")

        predicted_cls, conf = predict_lie_from_audio(test_video)

        if predicted_cls != -1:
            label = "Minciună" if predicted_cls == 1 else "Adevăr"
            print(f"\n--- Rezultat Predicție ---")
            print(f"Videoclip: {os.path.basename(test_video)}")
            print(f"Predicție: {label}")
            print(f"Încredere: {conf:.4f}")
        else:
            print("\n--- Predicția nu a putut fi realizată ---")