import os
import traceback
import shutil
import uuid

from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from sqlalchemy.orm import Session
from models import Video
from database import SessionLocal
from auth.jwt_bearer import get_current_user
from models import User

from ai_model.face_analysis.face_processor_3d import predict_lie_detection as predict_face_3d
from ai_model.face_analysis.face_processor2d import predict_lie_detection as predict_face
from ai_model.voice_analysis.voice_processor import predict_lie_from_audio as predict_voice
from ai_model.model_combiner import combine_predictions_fuzzy

router = APIRouter(
    prefix="/upload",
    tags=["upload"]
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/")
async def upload_video(
    video: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    file_ext = video.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{file_ext}"
    save_path = f"uploaded_videos/{filename}"

    os.makedirs("uploaded_videos", exist_ok=True)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    try:
        # 1. Obține predicția facială și vocală
        predicted_class, confidence = predict_face_3d(save_path)
        confidence = confidence * 100
       # voice_pred, voice_conf = predict_voice(save_path)
       # predicted_class, confidence = combine_predictions_fuzzy(face_pred, face_conf, voice_pred, voice_conf)

        # 3. Interpretează rezultatul
        if predicted_class == -1:
            prediction = "Incert"
        elif predicted_class == 1:
            prediction = "Fals"
        elif predicted_class == 0:
            prediction = "Adevărat"
        else:
            prediction = "Necunoscut"

        # 4. Salvează în baza de date
        video_entry = Video(
            filename=filename,
            prediction=prediction,
            confidence=float(confidence),
            user_id=current_user.id
        )
        db.add(video_entry)
        db.commit()
        db.refresh(video_entry)

        return {
            "filename": filename,
            "prediction": prediction,
            "confidence": int(confidence),
            "url": f"http://localhost:8000/videos/{filename}"
        }

    except Exception as e:
        print("Eroare la upload:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=traceback.format_exc())
