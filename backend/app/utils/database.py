import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.base import Base

APP_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = APP_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent

load_dotenv(BACKEND_DIR / '.env')
load_dotenv(PROJECT_DIR / '.env', override=False)


@lru_cache(maxsize=1)
def get_database_url() -> str:
    database_url = os.getenv('DATABASE_URL', '').strip()
    if not database_url:
        raise RuntimeError('DATABASE_URL is not configured.')
    return database_url


@lru_cache(maxsize=1)
def get_engine():
    return create_engine(get_database_url())


@lru_cache(maxsize=1)
def get_session_factory():
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


def get_db():
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=get_engine())