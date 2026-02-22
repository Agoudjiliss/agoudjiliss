"""
Auth Picture – Database Models
SQLAlchemy models for certified_images and audio_watermarks.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from auth_docs.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


class CertifiedImage(Base):
    """Stores metadata for each certified image."""
    __tablename__ = "certified_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    original_filename = Column(String(512), nullable=False)
    sha256_hash = Column(String(64), nullable=False, index=True)
    phash = Column(String(64), nullable=False, index=True)
    output_filename = Column(String(512), nullable=False)
    ai_score_1 = Column(Float, nullable=True)
    ai_score_2 = Column(Float, nullable=True)
    ai_label = Column(String(32), nullable=True)
    c2pa_signed = Column(Integer, default=0)  # 0=no, 1=yes
    robust_level = Column(String(16), default="medium")
    prompt_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AudioWatermark(Base):
    """Stores metadata for each enrolled audio watermark."""
    __tablename__ = "audio_watermarks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    original_filename = Column(String(512), nullable=False)
    output_filename = Column(String(512), nullable=False)
    payload_hex = Column(String(256), nullable=False, index=True)
    hmac_hex = Column(String(64), nullable=False)
    sample_rate = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# Engine & session factory
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency: yields a DB session and closes it afterward."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
