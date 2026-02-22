"""
Auth Picture – Image Authentication Route
POST /api/auth   → certify an image
GET  /api/download/{id} → download a certified image
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Query, Depends, HTTPException
from sqlalchemy.orm import Session

from auth_docs.config import (
    UPLOAD_DIR, OUTPUT_DIR, CERTS_DIR,
    WATERMARK_STRENGTH, ROBUST_LEVEL, PROMPT_TEXT,
    PGD_ENABLED, C2PA_CLAIM_GENERATOR,
)
from auth_docs.app.models.database import CertifiedImage, get_db
from auth_docs.app.services.hasher import sha256_file, phash_image
from auth_docs.app.services.protection import protect_image
from auth_docs.app.services.crypto import ensure_certs
from auth_docs.app.services.c2pa_service import sign_image
from auth_docs.app.services.ai_detector import detect_ai
from PIL import Image
from fastapi.responses import FileResponse

logger = logging.getLogger("auth_picture.routes.auth")
router = APIRouter()


@router.post("/api/auth")
async def certify_image(
    file: UploadFile = File(...),
    robust_level: Optional[str] = Query(None, description="low/medium/high"),
    prompt_text: Optional[str] = Query(None, description="Anti-AI prompt text"),
    db: Session = Depends(get_db),
):
    """Certify an image: AI detect, hash, watermark, C2PA sign, store in DB."""
    # Resolve parameters
    r_level = robust_level or ROBUST_LEVEL
    p_text = prompt_text if prompt_text is not None else PROMPT_TEXT

    # Ensure directories exist
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    upload_path = UPLOAD_DIR / file.filename
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Open image
    try:
        img = Image.open(upload_path)
        if img.mode not in ("L", "RGB", "RGBA"):
            img = img.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # AI detection (graceful – models may not be downloaded)
    ai_result = {"score_1": None, "score_2": None, "is_ai": False, "avg_score": 0.0}
    try:
        ai_result = detect_ai(img)
    except Exception as e:
        logger.warning("AI detection failed (models may not be available): %s", e)

    # Compute hashes
    sha_hash = sha256_file(upload_path)
    p_hash = phash_image(upload_path)

    # Protect image (watermark + DCT noise)
    try:
        protected_img = protect_image(
            img, sha_hash, p_hash,
            prompt_text=p_text,
            strength=WATERMARK_STRENGTH,
            robust_level=r_level,
        )
    except Exception as e:
        logger.error("Protection failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Image protection failed: {e}")

    # Save protected image (before C2PA, as C2PA reads from file)
    output_filename = f"certified_{file.filename}"
    # Ensure we save as PNG (lossless) for watermark integrity
    if not output_filename.lower().endswith(".png"):
        output_filename = Path(output_filename).stem + ".png"
    output_path = OUTPUT_DIR / output_filename
    protected_img.save(str(output_path), format="PNG")

    # C2PA signing
    c2pa_signed = 0
    try:
        cert_path, key_path = ensure_certs(CERTS_DIR)
        c2pa_output = OUTPUT_DIR / f"c2pa_{output_filename}"
        success = sign_image(
            output_path, c2pa_output,
            cert_path, key_path,
            claim_generator=C2PA_CLAIM_GENERATOR,
            prompt_text=p_text,
        )
        if success:
            # Replace output with C2PA-signed version
            shutil.move(str(c2pa_output), str(output_path))
            c2pa_signed = 1
    except Exception as e:
        logger.warning("C2PA signing failed: %s", e)

    # Store in database
    record = CertifiedImage(
        original_filename=file.filename,
        sha256_hash=sha_hash,
        phash=p_hash,
        output_filename=output_filename,
        ai_score_1=ai_result.get("score_1"),
        ai_score_2=ai_result.get("score_2"),
        ai_label="ai" if ai_result.get("is_ai") else "human",
        c2pa_signed=c2pa_signed,
        robust_level=r_level,
        prompt_text=p_text,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "id": record.id,
        "filename": output_filename,
        "sha256": sha_hash,
        "phash": p_hash,
        "ai_detection": ai_result,
        "c2pa_signed": bool(c2pa_signed),
        "robust_level": r_level,
        "prompt_text": p_text,
        "download_url": f"/api/download/{record.id}",
    }


@router.get("/api/download/{image_id}")
async def download_image(image_id: int, db: Session = Depends(get_db)):
    """Download a certified image by its ID."""
    record = db.query(CertifiedImage).filter(CertifiedImage.id == image_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Image not found")

    file_path = OUTPUT_DIR / record.output_filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(
        str(file_path),
        media_type="image/png",
        filename=record.output_filename,
    )
