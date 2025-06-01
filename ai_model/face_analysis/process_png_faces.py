import os
import cv2
import pandas as pd
from tqdm import tqdm

# CONFIG: modifică aceste căi după sistemul tău
lie_dir = r"C:\Users\Andrei\Desktop\Faculta\Lie Detection\Lie"
truth_dir = r"C:\Users\Andrei\Desktop\Faculta\Lie Detection\Truth"
output_dir_base = r"C:\Users\Andrei\Desktop\Faculta\Lie Detection\cropped_faces"
csv_output_path = r"C:\Users\Andrei\Desktop\Faculta\Lie Detection\preprocessed_data_from_png.csv"

# Inițializare clasificator pentru față
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
csv_rows = []

def detect_and_crop_face(img_path, output_path):
    img = cv2.imread(img_path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return False

    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, face)
    return True

def process_folder(input_dir, label_name, label_numeric):
    for filename in tqdm(os.listdir(input_dir), desc=f"Procesare {label_name}"):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(input_dir, filename)
            output_subdir = os.path.join(output_dir_base, os.path.splitext(filename)[0])
            output_img_path = os.path.join(output_subdir, "frame_00000.jpg")

            success = detect_and_crop_face(img_path, output_img_path)
            if success:
                csv_rows.append({
                    "id": filename,
                    "class": label_name,
                    "label_numeric": label_numeric
                })

# Execută procesarea
process_folder(lie_dir, "deceptive", 1)
process_folder(truth_dir, "truthful", 0)

# Scrie CSV
df = pd.DataFrame(csv_rows)
df.to_csv(csv_output_path, index=False)
print(f"CSV salvat în: {csv_output_path}")
