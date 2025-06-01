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
        frame_files = sorted(frame_files, key=lambda x: int(x.split('_')[1].split('.')[0]))

        for frame_file in tqdm(frame_files, desc=f"Procesare cadre din {video_folder}", leave=False):
            frame_path = os.path.join(input_folder_path, frame_file)
            detect_and_crop_face(frame_path, output_folder_path, face_cascade)

def predict_lie_detection(video_path, model_path="C:\\Users\\Andrei\\Desktop\\Faculta\\Lie Detection\\ai_model\\face_analysis\\face_2d_cnn_best.keras", img_height=64, img_width=64):
    """
    Realizează predicția de detectare a minciunilor pe un videoclip folosind un model 2D CNN.

    Args:
        video_path (str): Calea către fișierul videoclip.
        model_path (str): Calea către modelul antrenat.
        img_height (int): Înălțimea dorită a cadrelor.
        img_width (int): Lățimea dorită a cadrelor.

    Returns:
        tuple: Tuplu cu predicția (0 sau 1) și nivelul de încredere (probabilitatea).
    """

    # 1. Extrage cadrele din videoclip
    temp_frames_dir = "temp_frames"
    os.makedirs(temp_frames_dir, exist_ok=True)
    extract_frames(video_path, temp_frames_dir, frames_per_second=10)

    # 2. Detectează și decupează fețele din cadre
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise IOError('Nu s-a putut încărca clasificatorul Haar Cascade.')

    cropped_faces_dir = "temp_cropped_faces"
    os.makedirs(cropped_faces_dir, exist_ok=True)

    frame_files = [f for f in os.listdir(temp_frames_dir) if f.endswith('.jpg')]
    for frame_file in frame_files:
        frame_path = os.path.join(temp_frames_dir, frame_file)
        detect_and_crop_face(frame_path, cropped_faces_dir, face_cascade)

    # 3. Încarcă și preprocesează fețele decupate pentru modelul 2D
    face_frames = sorted([f for f in os.listdir(cropped_faces_dir) if f.endswith('.jpg')])

    X = []
    for face_frame in face_frames:
        face_path = os.path.join(cropped_faces_dir, face_frame)
        img = cv2.imread(face_path)
        img = cv2.resize(img, (img_width, img_height))
        img = img.astype(np.float32) / 255.0
        X.append(img)
    X = np.array(X)

    # 4. Încarcă modelul antrenat și face predicțiile pentru fiecare cadru
    model = load_model(model_path)
    predictions = model.predict(X)  # Obținem predicții pentru toate cadrele
    predicted_classes = np.argmax(predictions, axis=1)  # Clasele prezise
    confidences = np.max(predictions, axis=1)  # Nivelurile de încredere

    # 5. Agregă predicțiile (exemplu: majoritatea)
    # Aceasta este o metodă simplă. Poți folosi abordări mai sofisticate.
    if len(predicted_classes) > 0:
        final_predicted_class = np.argmax(np.bincount(predicted_classes))
        final_confidence = np.mean(confidences)  # Media nivelurilor de încredere
    else:
        final_predicted_class = -1  # Incert
        final_confidence = 0.5

    # 6. Curățăm folderele temporare
    shutil.rmtree(temp_frames_dir)
    shutil.rmtree(cropped_faces_dir)

    return final_predicted_class, final_confidence

if __name__ == '__main__':

    process_extracted_frames()

