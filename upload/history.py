from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
from models import Video, User
from auth.jwt_bearer import get_current_user
from fastapi.responses import JSONResponse

router = APIRouter(
    prefix="/history",
    tags=["History"]
)



@router.get("/")
def get_user_videos(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    videos = db.query(Video).filter(Video.user_id == current_user.id).all()

    return [
        {
            "id": video.id,
            "prediction": video.prediction,
            "confidence": video.confidence,
            "url": f"http://localhost:8000/videos/{video.filename}"
        }
        for video in videos
    ]
