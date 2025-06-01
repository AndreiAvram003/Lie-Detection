import cv2
import os
import numpy as np
import pandas as pd
from keras.src.saving import load_model
from tqdm import tqdm
from ai_model.face_analysis.face_extractor import extract_frames  # Importăm funcția de extragere a cadrelor
import shutil
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def detect_and_crop_face(image_path, output_dir, face_cascade):
    """
    Detectează și decupează fața dintr-o imagine, salvând fața decupată.

    Args:
        image_path (str): Calea către imagine.
        output_dir (str): Calea către directorul unde va fi salvată fața decupată.
        face_cascade (cv2.CascadeClassifier): Obiectul clasificatorului Haar Cascade pentru detectarea fețelor.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Eroare: Nu s-a putut citi imaginea: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 1:
        (x, y, w, h) = faces[0]
        face_roi = img[y:y + h, x:x + w]
        output_filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, face_roi)
        return output_path
    elif len(faces) > 1:
        print(f"Avertisment: S-au detectat mai multe fețe în {image_path}. Se va salva doar prima față detectată.")
        (x, y, w, h) = faces[0]
        face_roi = img[y:y + h, x:x + w]
        output_filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, face_roi)
        return output_path
    else:
        #print(f"Avertisment: Nicio față detectată în {image_path}.")
        return None

def process_extracted_frames(input_base_dir="extracted_frames", output_base_dir="cropped_faces"):
    """
    Procesează toate cadrele extrase, detectând și decupând fețele din fiecare.

    Args:
        input_base_dir (str): Directorul de bază unde se află cadrele extrase.
        output_base_dir (str): Directorul de bază unde vor fi salvate fețele decupate.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise IOError('Nu s-a putut încărca clasificatorul Haar Cascade.')

    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    video_folders = [f for f in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, f))]

    for video_folder in tqdm(video_folders, desc="Procesare foldere video"):
        input_folder_path = os.path.join(input_base_dir, video_folder)
        output_folder_path = os.path.join(output_base_dir, video_folder)
        os.makedirs(output_folder_path, exist_ok=True)

        frame_files = [f for f in os.listdir(input_folder_path) if os.path.isfile(os.path.join(input_folder_path, f))]

        for frame_file in tqdm(frame_files, desc=f"Procesare cadre din {video_folder}", leave=False):
            frame_path = os.path.join(input_folder_path, frame_file)
            detect_and_crop_face(frame_path, output_folder_path, face_cascade)

def predict_lie_detection(video_path, model_path="C:\\Users\\Andrei\\Desktop\\Faculta\\Lie Detection\\ai_model\\face_analysis\\face_3d_cnn_best.keras", img_height=64, img_width=64):
    """
    Realizează predicția de detectare a minciunilor pe un videoclip folosind un model 3D CNN.

    Args:
        video_path (str): Calea către fișierul videoclip.
        model_path (str): Calea către modelul antrenat.
        img_height (int): Înălțimea cadrelor.
        img_width (int): Lățimea cadrelor.

    Returns:
        tuple: (predicted_class, confidence_score)
    """

    REQUIRED_FRAMES = 16

    # 1. Extrage cadrele din video
    temp_frames_dir = "temp_frames"
    os.makedirs(temp_frames_dir, exist_ok=True)
    extract_frames(video_path, temp_frames_dir, frames_per_second=10)

    # 2. Detectează și decupează fețele

    cropped_faces_dir = "temp_cropped_faces"
    os.makedirs(cropped_faces_dir, exist_ok=True)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise IOError('Nu s-a putut încărca clasificatorul Haar Cascade.')

    frame_files = sorted([f for f in os.listdir(temp_frames_dir) if f.endswith('.jpg')])
    frame_files = sorted(frame_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    for frame_file in frame_files:
        frame_path = os.path.join(temp_frames_dir, frame_file)
        detect_and_crop_face(frame_path, cropped_faces_dir,face_cascade)

    face_frames = sorted([f for f in os.listdir(cropped_faces_dir) if f.endswith('.jpg')])
    if len(face_frames) < REQUIRED_FRAMES:
        raise ValueError(f"Videoclipul are doar {len(face_frames)} cadre — sunt necesare minim {REQUIRED_FRAMES} pentru modelul 3D.")

    # 3. Preprocesare imagini
    selected_frames = face_frames[:REQUIRED_FRAMES]
    X = []
    for frame in selected_frames:
        img = cv2.imread(os.path.join(cropped_faces_dir, frame))
        img = cv2.resize(img, (img_width, img_height))
        X.append(img)

    X = np.array(X, dtype=np.float32) / 255.0  # Normalizare
    X = np.expand_dims(X, axis=0)  # (1, 16, 64, 64, 3)

    # 4. Predictie
    model = load_model(model_path)
    prediction = model.predict(X)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    # 5. Curățare
    shutil.rmtree(temp_frames_dir)
    shutil.rmtree(cropped_faces_dir)

    return predicted_class, confidence

if __name__ == '__main__':
    # Exemplu de utilizare
    video_path = 'cale/catre/videoclipul_tau.mp4'  # Înlocuiește cu calea reală
    predicted_class, confidence = predict_lie_detection(video_path)

    if predicted_class == 1:
        print("Predicție: Fals")
    elif predicted_class == 0:
        print("Predicție: Adevărat")
    else:
        print("Predicție: Incertă")
    print(f"Nivel de încredere: {confidence * 100:.2f}%")

