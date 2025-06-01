from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal
from schemas import UserCreate, ShowUser
from models import User
from auth.hashing import hash_password, verify_password
from auth.jwt_handler import create_access_token
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from auth.jwt_bearer import get_current_user


router = APIRouter(
    prefix="/auth",
    tags=["auth"]
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal
from schemas import UserCreate, ShowUser
from models import User
from auth.hashing import hash_password
from auth.jwt_handler import create_access_token
from pydantic import BaseModel

router = APIRouter(
    prefix="/auth",
    tags=["auth"]
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Modificăm ruta de înregistrare să verifice și username-ul, nu doar email-ul
@router.post("/register", response_model=ShowUser)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user_by_email = db.query(User).filter(User.email == user.email).first()
    db_user_by_username = db.query(User).filter(User.username == user.username).first()

    # Verifică dacă email-ul sau username-ul există deja
    if db_user_by_email:
        raise HTTPException(status_code=400, detail="Email deja înregistrat")
    if db_user_by_username:
        raise HTTPException(status_code=400, detail="Username deja înregistrat")

    hashed_pw = hash_password(user.password)
    new_user = User(username=user.username, email=user.email, password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user



class LoginRequest(BaseModel):
    email_or_username: str
    password: str

@router.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    # Căutăm utilizatorul după email sau username
    user = db.query(User).filter(
        (User.email == request.email_or_username) | (User.username == request.email_or_username)
    ).first()

    if not user or not verify_password(request.password, user.password):
        raise HTTPException(status_code=400, detail="Credențiale invalide")

    token = create_access_token({"sub": user.email})
    return {"access_token": token, "token_type": "bearer"}


@router.post("/logout")
def logout(current_user: User = Depends(get_current_user)):
    # Pur și simplu întoarcem un răspuns de succes. Pe client, se va șterge token-ul
    return {"message": "Logout successful"}