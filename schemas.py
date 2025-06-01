from pydantic import BaseModel, EmailStr
from typing import List, Optional

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class ShowUser(BaseModel):
    id: int
    username: str
    email: EmailStr
    class Config:
        orm_mode = True

class VideoCreate(BaseModel):
    filename: str
    prediction: str
    confidence: float

class ShowVideo(BaseModel):
    id: int
    filename: str
    prediction: str
    confidence: float
    class Config:
        orm_mode = True
