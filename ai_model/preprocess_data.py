import pandas as pd
import os

def load_and_preprocess_data(csv_path, video_base_path_deceptive, video_base_path_truthful):
    """
    Încarcă și preprocesează datele din fișierul CSV, adăugând calea către fișierul video.

    Args:
        csv_path (str): Calea către fișierul CSV.
        video_base_path_deceptive (str): Calea de bază către folderul cu videoclipuri deceptive.
        video_base_path_truthful (str): Calea de bază către folderul cu videoclipuri truthful.

    Returns:
        pandas.DataFrame: Un DataFrame cu datele preprocesate, incluzând calea către video.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Eroare: Fișierul CSV nu a fost găsit la calea: {csv_path}")
        return None

    video_paths = []
    for index, row in df.iterrows():
        video_id = row['id']
        label = row['class']
        video_filename = video_id  # Presupunem că 'id' conține numele fișierului video

        if label == 'deceptive':
            video_path = os.path.join(video_base_path_deceptive, video_filename)
        elif label == 'truthful':
            video_path = os.path.join(video_base_path_truthful, video_filename)
        else:
            video_path = None
            print(f"Avertisment: Etichetă necunoscută '{label}' pentru {video_id}")

        if video_path and os.path.exists(video_path):
            video_paths.append(video_path)
        else:
            video_paths.append(None)
            print(f"Avertisment: Fișierul video nu a fost găsit la calea: {video_path} pentru {video_id}")

    df['video_path'] = video_paths
    df = df.dropna(subset=['video_path']) # Elimină rândurile pentru care nu s-a găsit calea video

    # Poți adăuga aici și alte preprocesări ale datelor din CSV dacă este necesar
    # De exemplu, codificarea etichetelor textuale ('deceptive'/'truthful') în valori numerice (0/1)
    df['label_numeric'] = df['class'].apply(lambda x: 1 if x == 'deceptive' else 0)

    return df

if __name__ == '__main__':
    # Căile către fișierul tău CSV și folderele cu videoclipuri
    csv_file_path = r'C:\Users\Andrei\Desktop\Faculta\Lie Detection\Real-life_Deception_Detection_2016\Annotation\All_Gestures_Deceptive and Truthful.csv'
    deceptive_videos_path = r'C:\Users\Andrei\Desktop\Faculta\Lie Detection\Real-life_Deception_Detection_2016\Clips\Deceptive'
    truthful_videos_path = r'C:\Users\Andrei\Desktop\Faculta\Lie Detection\Real-life_Deception_Detection_2016\Clips\Truthful'

    processed_df = load_and_preprocess_data(csv_file_path, deceptive_videos_path, truthful_videos_path)

    if processed_df is not None:
        print("Date preprocesate cu succes:")
        print(processed_df.head())

        # Salvează DataFrame-ul preprocesat într-un fișier CSV
        output_csv_path = 'preprocessed_data.csv'
        processed_df.to_csv(output_csv_path, index=False)
        print(f"\nDataFrame-ul preprocesat a fost salvat în: {output_csv_path}")