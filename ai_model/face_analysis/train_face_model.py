import os
import cv2
import numpy as np
import pandas as pd
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from  cnn_2d_model import create_2d_cnn_model  # Importăm modelul 2D
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.utils import compute_class_weight


def load_and_preprocess_data_2d(data_dir, df, img_height, img_width):
    """Încarcă și preprocesează *fiecare cadru individual* pentru antrenare cu model 2D."""

    X = []
    y = []
    for index, row in df.iterrows():
        video_id = row['id'].replace('.mp4', '')
        label = row['label_numeric']
        video_dir = os.path.join(data_dir, video_id)
        if not os.path.exists(video_dir):
            print(f"Avertisment: Directorul {video_dir} nu există. Se sare peste.")
            continue

        frames = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg') ])
        for frame in frames:  # Procesăm fiecare cadru individual
            frame_path = os.path.join(video_dir, frame)
            img = cv2.imread(frame_path)
            img = cv2.resize(img, (img_width, img_height))
            X.append(img)
            y.append(label)

    X = np.array(X, dtype=np.float32) / 255.0
    y = to_categorical(y)
    return X, y

if __name__ == '__main__':
    # Configurare
    data_dir = 'cropped_faces'
    csv_file = '../final_data.csv'
    img_height = 64
    img_width = 64
    num_classes = 2
    batch_size = 32
    epochs = 30

    # Încărcăm și preprocesăm datele pentru modelul 2D
    df = pd.read_csv(csv_file)
    X, y = load_and_preprocess_data_2d(data_dir, df, img_height, img_width)

    # Împărțim datele
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creăm modelul 2D
    input_shape = (img_height, img_width, 3)
    model = create_2d_cnn_model(input_shape, num_classes)

    # Compilăm modelul
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Callback-uri
    checkpoint = ModelCheckpoint('face_2d_cnn_best.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)


    data_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train_generator = data_gen.flow(X_train, y_train, batch_size=batch_size)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(np.argmax(y_train, axis=1)),
        y=np.argmax(y_train, axis=1)
    )
    class_weight_dict = dict(enumerate(class_weights))

    history = model.fit(
        train_generator,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        class_weight=class_weight_dict
    )

    # Vizualizare
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

    # Evaluare
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Validation Set)')
    plt.show()
    print(classification_report(y_true, y_pred_classes))