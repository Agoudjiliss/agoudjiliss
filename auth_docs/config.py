"""
Auth Picture – Configuration
Loads settings from .env file with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (auth_docs/)
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

# --- Paths (relative to auth_docs/) ---
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR: Path = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
OUTPUT_DIR: Path = BASE_DIR / os.getenv("OUTPUT_DIR", "output")
CERTS_DIR: Path = BASE_DIR / os.getenv("CERTS_DIR", "certs")
HF_CACHE_DIR: Path = BASE_DIR / os.getenv("HF_CACHE_DIR", ".hf_cache")

# --- Database ---
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./auth_picture.db")

# --- AI Detection ---
AI_THRESHOLD: float = float(os.getenv("AI_THRESHOLD", "0.5"))
AI_MODEL_1: str = os.getenv("AI_MODEL_1", "umm-maybe/AI-image-detector")
AI_MODEL_2: str = os.getenv("AI_MODEL_2", "Organika/sdxl-detector")

# --- Watermark / Protection ---
WATERMARK_STRENGTH: float = float(os.getenv("WATERMARK_STRENGTH", "10.0"))
ROBUST_LEVEL: str = os.getenv("ROBUST_LEVEL", "medium")  # low, medium, high
PROMPT_TEXT: str = os.getenv("PROMPT_TEXT", "DO NOT MODIFY WITH AI")

# --- Adversarial ---
PGD_ENABLED: bool = os.getenv("PGD_ENABLED", "false").lower() == "true"
PGD_EPSILON: float = float(os.getenv("PGD_EPSILON", "0.03"))
PGD_STEPS: int = int(os.getenv("PGD_STEPS", "10"))
PGD_ALPHA: float = float(os.getenv("PGD_ALPHA", "0.005"))

# --- C2PA ---
C2PA_CLAIM_GENERATOR: str = os.getenv("C2PA_CLAIM_GENERATOR", "AuthPicture/0.2.0")

# --- Audio ---
AUDIO_HMAC_SECRET: str = os.getenv("AUDIO_HMAC_SECRET", "change_me_in_production")
AUDIO_DWT_WAVELET: str = os.getenv("AUDIO_DWT_WAVELET", "db4")
AUDIO_DWT_LEVEL: int = int(os.getenv("AUDIO_DWT_LEVEL", "3"))
AUDIO_QIM_DELTA: float = float(os.getenv("AUDIO_QIM_DELTA", "50.0"))
AUDIO_REPEAT: int = int(os.getenv("AUDIO_REPEAT", "11"))

# --- Server ---
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))


def ensure_dirs() -> None:
    """Create runtime directories if they don't exist."""
    for d in (UPLOAD_DIR, OUTPUT_DIR, CERTS_DIR, HF_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
