import os
import shutil

import cv2
import numpy as np
import pandas as pd
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from cnn3d_model import create_3d_cnn_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.utils import compute_class_weight
from tensorflow.keras.utils import Sequence


def load_sequences(data_dir, df, num_frames, height, width):
    X = []
    y = []

    for index, row in df.iterrows():
        video_id = row['id'].replace('.mp4', '')
        label = row['label_numeric']
        video_path = os.path.join(data_dir, video_id)

        if not os.path.exists(video_path):
            print(f"Director inexistent: {video_path}")
            continue

        frames = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
        if len(frames) < num_frames:
            continue
        for i in range(0, len(frames) - num_frames + 1, num_frames):
            clip = []
            for j in range(i, i + num_frames):
                img = cv2.imread(os.path.join(video_path, frames[j]))
                img = cv2.resize(img, (width, height))
                clip.append(img)
            X.append(clip)
            y.append(label)

    X = np.array(X, dtype=np.float32) / 255.0
    y = to_categorical(y)
    return X, y






class Augment3DSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, image_data_generator_args, shuffle=True):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        # Folosim ImageDataGenerator din importul tău existent
        self.img_data_gen = ImageDataGenerator(**image_data_generator_args)
        self.shuffle = shuffle
        self.indices = np.arange(len(self.x))
        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        # Numărul de batch-uri per epocă
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        # Generează un batch de date
        # Obține indicii pentru batch-ul curent
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Preia datele originale pentru batch
        batch_x_orig = self.x[batch_indices]
        batch_y = self.y[batch_indices]

        # Inițializează un array pentru datele augmentate din batch
        batch_x_aug = np.zeros_like(batch_x_orig, dtype=np.float32)

        # Aplică augmentarea 2D pe fiecare frame din fiecare secvență din batch
        for i in range(batch_x_orig.shape[0]):  # Iterează peste eșantioanele din batch
            for j in range(batch_x_orig.shape[1]):  # Iterează peste frame-urile din secvență
                frame = batch_x_orig[i, j, :, :, :]  # Extrage un frame 2D
                # Aplică o transformare aleatorie definită în ImageDataGenerator
                augmented_frame = self.img_data_gen.random_transform(frame)
                batch_x_aug[i, j, :, :, :] = augmented_frame

        return batch_x_aug, batch_y

    def on_epoch_end(self):
        # Amestecă indicii la sfârșitul fiecărei epoci, dacă este activat
        if self.shuffle:
            np.random.shuffle(self.indices)


if __name__ == '__main__':
    # Config
    data_dir = 'cropped_faces'
    csv_file = '../final_data.csv'
    height, width = 64, 64
    num_frames = 16
    num_classes = 2
    batch_size = 8
    epochs = 50

    df = pd.read_csv(csv_file)
    X, y = load_sequences(data_dir, df, num_frames, height, width)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    augmentation_args = dict(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest' # Adaugă fill_mode pentru a gestiona pixelii noi
    )
    train_generator = Augment3DSequence(X_train, y_train, batch_size, augmentation_args, shuffle=True)

    input_shape = (num_frames, height, width, 3)
    model = create_3d_cnn_model(input_shape, num_classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    checkpoint = ModelCheckpoint('face_3d_cnn_best.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)


    data_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )


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

    # Grafic performanță
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title("Loss")
    plt.legend()
    plt.show()

    # Matrice de confuzie
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Validation Set)")
    plt.show()
    print(classification_report(y_true, y_pred_classes))
