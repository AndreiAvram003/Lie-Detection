import cv2
import os
import shutil
import mediapipe as mp  # Importăm MediaPipe
import numpy as np

# --- CONFIGURARE ---
# !!! ACTUALIZEAZĂ ACESTE CĂI ÎNAINTE DE A RUCA SCRIPTUL !!!
SOURCE_EXTRACTED_FRAMES_DIR = r"C:\Users\Andrei\Desktop\Faculta\Lie Detection\ai_model\face_analysis\extracted_frames"  # Directorul cu frame-urile originale
TARGET_NEW_CROPPED_FACES_DIR = r"C:\Users\Andrei\Desktop\Faculta\Lie Detection\ai_model\face_analysis\new_cropped_faces"  # Noul director pentru fețele decupate cu MediaPipe

# Dimensiunea la care vor fi redimensionate fețele decupate
OUTPUT_CROP_WIDTH = 64
OUTPUT_CROP_HEIGHT = 64

# Pragul minim de încredere pentru detecțiile MediaPipe
MIN_DETECTION_CONFIDENCE = 0.5
# --- SFÂRȘIT CONFIGURARE ---

# Inițializare MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection


# mp_drawing = mp.solutions.drawing_utils # Nu este necesar pentru decupare, doar pentru vizualizare

def process_frames_with_mediapipe():
    """
    Parcurge frame-urile extrase, detectează fețele folosind MediaPipe,
    le decupează și le salvează.
    """
    if not os.path.exists(SOURCE_EXTRACTED_FRAMES_DIR):
        print(f"[EROARE] Directorul sursă '{SOURCE_EXTRACTED_FRAMES_DIR}' nu există!")
        return

    os.makedirs(TARGET_NEW_CROPPED_FACES_DIR, exist_ok=True)
    print(f"Se procesează frame-urile din: '{SOURCE_EXTRACTED_FRAMES_DIR}' folosind MediaPipe")
    print(f"Fețele decupate vor fi salvate în: '{TARGET_NEW_CROPPED_FACES_DIR}'\n")

    processed_videos_count = 0
    processed_frames_count = 0
    faces_found_count = 0

    # Inițializează detectorul de fețe MediaPipe
    # model_selection=0 este pentru fețe apropiate (sub 2m), model_selection=1 este pentru distanțe mai mari (până la 5m).
    # Alegem 1 pentru robustețe generală.
    with mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE) as face_detector:

        for video_dir_name in os.listdir(SOURCE_EXTRACTED_FRAMES_DIR):
            source_video_subdir_path = os.path.join(SOURCE_EXTRACTED_FRAMES_DIR, video_dir_name)

            if os.path.isdir(source_video_subdir_path):
                target_video_subdir_path = os.path.join(TARGET_NEW_CROPPED_FACES_DIR, video_dir_name)
                os.makedirs(target_video_subdir_path, exist_ok=True)

                print(f"Procesare director videoclip: '{video_dir_name}'")
                current_video_frames_processed = 0
                current_video_faces_found = 0

                for frame_filename in sorted(os.listdir(source_video_subdir_path)):
                    if frame_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        source_frame_full_path = os.path.join(source_video_subdir_path, frame_filename)
                        target_frame_full_path = os.path.join(target_video_subdir_path, frame_filename)

                        current_video_frames_processed += 1

                        image = cv2.imread(source_frame_full_path)
                        if image is None:
                            # print(f"  -> Avertisment: Nu s-a putut citi frame-ul '{source_frame_full_path}'. Se sare peste.")
                            continue

                        # MediaPipe se așteaptă la imagini RGB
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # Procesează imaginea și detectează fețele
                        results = face_detector.process(rgb_image)

                        if results.detections:
                            # Selectează cea mai bună detecție (cea cu scorul cel mai mare)
                            # MediaPipe returnează detecțiile sortate sau putem sorta manual
                            best_detection = sorted(results.detections, key=lambda det: det.score[0], reverse=True)[0]

                            img_h, img_w, _ = image.shape
                            bbox_data = best_detection.location_data.relative_bounding_box

                            if not all(hasattr(bbox_data, attr) for attr in ['xmin', 'ymin', 'width', 'height']):
                                # print(f"  -> Avertisment: Date bounding box incomplete (MediaPipe) pentru '{frame_filename}'. Se sare peste.")
                                continue

                            # Denormalizează coordonatele bounding box-ului
                            x_min_rel = bbox_data.xmin
                            y_min_rel = bbox_data.ymin
                            width_rel = bbox_data.width
                            height_rel = bbox_data.height

                            x_min_abs = int(x_min_rel * img_w)
                            y_min_abs = int(y_min_rel * img_h)
                            width_abs = int(width_rel * img_w)
                            height_abs = int(height_rel * img_h)

                            # Asigură coordonatele în limitele imaginii
                            x_max_abs = min(x_min_abs + width_abs, img_w - 1)
                            y_max_abs = min(y_min_abs + height_abs, img_h - 1)
                            x_min_abs = max(x_min_abs, 0)
                            y_min_abs = max(y_min_abs, 0)

                            # Calculează lățimea și înălțimea efective după constrângere
                            w_final = x_max_abs - x_min_abs
                            h_final = y_max_abs - y_min_abs

                            if w_final > 0 and h_final > 0:
                                face_roi = image[y_min_abs:y_max_abs, x_min_abs:x_max_abs]

                                if face_roi.size == 0:
                                    # print(f"  -> Avertisment: Decupajul feței (MediaPipe) este gol pentru '{frame_filename}'. Se sare peste.")
                                    continue

                                resized_face = cv2.resize(face_roi, (OUTPUT_CROP_WIDTH, OUTPUT_CROP_HEIGHT))
                                cv2.imwrite(target_frame_full_path, resized_face)
                                current_video_faces_found += 1
                            else:
                                # print(f"  -> Avertisment: Coordonate invalide pentru decupaj (MediaPipe) în '{frame_filename}'. Se sare peste.")
                                pass
                        else:
                            # print(f"  -> Avertisment: Nicio față detectată (MediaPipe) în '{frame_filename}'.")
                            pass

                print(
                    f"  => Finalizat pentru '{video_dir_name}': {current_video_faces_found} fețe găsite din {current_video_frames_processed} frame-uri procesate.")
                processed_videos_count += 1
                processed_frames_count += current_video_frames_processed
                faces_found_count += current_video_faces_found

    print(f"\n--- Procesarea completă (MediaPipe) ---")
    print(f"Total videoclipuri (directoare) procesate: {processed_videos_count}")
    print(f"Total frame-uri citite: {processed_frames_count}")
    print(f"Total fețe detectate și salvate: {faces_found_count}")
    print(f"Verifică rezultatele în: '{TARGET_NEW_CROPPED_FACES_DIR}'")


if __name__ == '__main__':
    process_frames_with_mediapipe()