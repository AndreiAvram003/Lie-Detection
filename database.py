import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

# Încarcă fișierul .env corespunzător
if os.getenv("TESTING") == "0":
    load_dotenv("C:\\Users\\Andrei\\Desktop\\Faculta\\Lie Detection\\.env")
else:
    load_dotenv("C:\\Users\\Andrei\\Desktop\\Faculta\\Lie Detection\\.env.test")

# Citește URL-ul bazei de date din variabilele de mediu
DATABASE_URL = os.getenv("DATABASE_URL")

# Creează engine-ul și sesiunea
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Baza pentru modelele ORM
Base = declarative_base()

# Funcție pentru obținerea sesiunii de DB
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
