"""
Auth Picture – Image Verification Route
POST /api/check → verify if an image is authentic
"""

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from auth_docs.config import UPLOAD_DIR
from auth_docs.app.models.database import CertifiedImage, get_db
from auth_docs.app.services.hasher import sha256_file, phash_image, phash_distance
from auth_docs.app.services.c2pa_service import verify_image

logger = logging.getLogger("auth_picture.routes.check")
router = APIRouter()

PHASH_THRESHOLD = 15  # Maximum Hamming distance for pHash similarity


@router.post("/api/check")
async def check_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Verify an image: C2PA check + hash comparison against DB.

    Verdict:
        - AUTHENTIC: exact SHA-256 match in DB
        - MODIFIED: pHash match (distance < 15) but SHA differs
        - UNKNOWN: no match found
    """
    # Save uploaded file
    upload_path = UPLOAD_DIR / f"check_{file.filename}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Compute hashes
    sha_hash = sha256_file(upload_path)
    p_hash = phash_image(upload_path)

    # C2PA verification
    c2pa_result = {"valid": False, "manifest": None, "error": None}
    try:
        c2pa_result = verify_image(upload_path)
    except Exception as e:
        logger.warning("C2PA verification failed: %s", e)
        c2pa_result["error"] = str(e)

    # Check exact SHA-256 match
    exact_match = db.query(CertifiedImage).filter(
        CertifiedImage.sha256_hash == sha_hash
    ).first()

    if exact_match:
        return {
            "verdict": "AUTHENTIC",
            "sha256_match": True,
            "phash_match": True,
            "phash_distance": 0,
            "c2pa_valid": c2pa_result["valid"],
            "matched_id": exact_match.id,
            "original_filename": exact_match.original_filename,
        }

    # Check pHash similarity
    all_records = db.query(CertifiedImage).all()
    best_match = None
    best_distance = float("inf")

    for record in all_records:
        try:
            dist = phash_distance(p_hash, record.phash)
            if dist < best_distance:
                best_distance = dist
                best_match = record
        except Exception:
            continue

    if best_match and best_distance < PHASH_THRESHOLD:
        return {
            "verdict": "MODIFIED",
            "sha256_match": False,
            "phash_match": True,
            "phash_distance": best_distance,
            "c2pa_valid": c2pa_result["valid"],
            "matched_id": best_match.id,
            "original_filename": best_match.original_filename,
        }

    return {
        "verdict": "UNKNOWN",
        "sha256_match": False,
        "phash_match": False,
        "phash_distance": best_distance if best_match else None,
        "c2pa_valid": c2pa_result["valid"],
        "matched_id": None,
        "original_filename": None,
    }
