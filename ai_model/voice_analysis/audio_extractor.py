import os
import pandas as pd
from moviepy import VideoFileClip
from pathlib import Path
from tqdm import tqdm

def extract_audio_from_videos(csv_path: str, output_dir: str = "audio_clips"):
    # Încarcă CSV-ul care conține coloana `video_path`
    df = pd.read_csv(csv_path)

    # Creează directorul de ieșire dacă nu există
    os.makedirs(output_dir, exist_ok=True)

    audio_paths = []

    print("Extragem audio din videoclipuri...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_path = row['video_path']
        video_name = Path(video_path).stem
        audio_path = os.path.join(output_dir, f"{video_name}.wav")

        try:
            clip = VideoFileClip(video_path)
            if clip.audio is None:
                print(f"[Avertisment] Nu s-a găsit audio pentru: {video_path}")
                continue
            clip.audio.write_audiofile(audio_path, logger=None)
            audio_paths.append(audio_path)
        except Exception as e:
            print(f"[Eroare] Nu am putut procesa {video_path} — {e}")

    print(f"✅ Audio extras pentru {len(audio_paths)} videoclipuri.")
    return audio_paths

# Exemplu de utilizare
if __name__ == "__main__":
    extract_audio_from_videos("C:\\Users\\Andrei\\Desktop\\Faculta\\Lie Detection\\ai_model\\preprocessed_data.csv")
