"""
Auth Picture – Main Application
FastAPI entry point with startup initialization, health check, and route mounting.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from auth_docs.config import ensure_dirs
from auth_docs.app.models.database import init_db
from auth_docs.app.routes.auth_picture import router as auth_router
from auth_docs.app.routes.check_picture import router as check_router
from auth_docs.app.routes.audio import router as audio_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("auth_picture")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create directories and initialize database."""
    logger.info("Auth Picture v0.2.0 starting up...")
    ensure_dirs()
    init_db()
    logger.info("Database initialized, directories ready.")
    yield
    logger.info("Auth Picture shutting down.")


app = FastAPI(
    title="Auth Picture API",
    description=(
        "API REST for authenticating and protecting multimedia documents "
        "(images & audio) against AI-generated falsifications."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# Mount routes
app.include_router(auth_router, tags=["Images"])
app.include_router(check_router, tags=["Verification"])
app.include_router(audio_router, tags=["Audio"])


@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "0.2.0",
        "project": "Auth Picture",
        "description": "Image & Audio Authentication API",
    }
