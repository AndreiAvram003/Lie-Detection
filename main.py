from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import random
import os
os.environ["TESTING"] = "0"
from fastapi.middleware.cors import CORSMiddleware
from database import Base, engine
from auth.auth_router import router as auth_router
from fastapi.staticfiles import StaticFiles
from upload.upload_router import router as upload_router
from upload.history import router as history_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Base.metadata.create_all(bind=engine)
app.include_router(auth_router)
app.include_router(upload_router)
app.include_router(history_router)

app.mount("/videos", StaticFiles(directory="C:\\Users\\Andrei\\Desktop\\Faculta\\Lie Detection\\uploaded_videos"), name="videos")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


