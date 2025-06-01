import cv2
import os
import pandas as pd
from tqdm import tqdm  # Pentru bara de progres

def extract_frames(video_path, output_dir, frames_per_second=10):
    """
    Extrage cadre dintr-un videoclip la o anumită rată și le salvează într-un director.

    Args:
        video_path (str): Calea către fișierul videoclip.
        output_dir (str): Calea către directorul unde vor fi salvate cadrele.
        frames_per_second (int): Numărul de cadre de extras pe secundă.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Eroare: Nu s-a putut deschide videoclipul: {video_path}")
        return

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extract_every = int(frame_rate / frames_per_second)

    count = 0
    extracted_count = 0
    with tqdm(total=total_frames, desc=f"Extragere cadre din {os.path.basename(video_path)}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % extract_every == 0:
                frame_filename = f"frame_{extracted_count:05d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                extracted_count += 1

            count += 1
            pbar.update(1)

    cap.release()
    print(f"S-au extras {extracted_count} cadre din {os.path.basename(video_path)} și au fost salvate în {output_dir}")

def process_videos_for_frames(df, output_base_dir="extracted_frames", frames_per_second=10):
    """
    Procesează un DataFrame cu căi către videoclipuri pentru a extrage cadre din fiecare.

    Args:
        df (pandas.DataFrame): DataFrame cu o coloană 'video_path' și o coloană 'id'.
        output_base_dir (str): Directorul de bază unde vor fi create subdirectoare pentru fiecare videoclip.
        frames_per_second (int): Numărul de cadre de extras pe secundă.
    """
    for index, row in df.iterrows():
        video_path = row['video_path']
        video_id = row['id']
        output_dir = os.path.join(output_base_dir, video_id.replace('.mp4', '')) # Creează un folder cu numele ID-ului video
        extract_frames(video_path, output_dir, frames_per_second)

if __name__ == '__main__':
    # Calea către fișierul CSV preprocesat
    processed_csv_path = '../preprocessed_data.csv'  # Presupunem că este în directorul principal

    if os.path.exists(processed_csv_path):
        try:
            df = pd.read_csv(processed_csv_path)
            process_videos_for_frames(df)
        except FileNotFoundError:
            print(f"Eroare: Fișierul CSV preprocesat nu a fost găsit la: {processed_csv_path}")
        except Exception as e:
            print(f"Eroare la citirea sau procesarea fișierului CSV: {e}")
    else:
        print(f"Eroare: Fișierul CSV preprocesat nu există la calea: {processed_csv_path}")