"""
Auth Picture – Audio Routes
POST /api/audio/enroll   → enroll (watermark) an audio file
POST /api/audio/verify   → verify an audio file's watermark
GET  /api/audio/download/{id} → download an enrolled audio file
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from auth_docs.config import (
    UPLOAD_DIR, OUTPUT_DIR,
    AUDIO_HMAC_SECRET, AUDIO_DWT_WAVELET, AUDIO_DWT_LEVEL,
    AUDIO_QIM_DELTA, AUDIO_REPEAT,
)
from auth_docs.app.models.database import AudioWatermark, get_db
from auth_docs.app.services.audio_watermark import (
    generate_audio_payload, read_wav, write_wav,
    embed_audio_watermark, extract_audio_watermark,
    frame_audio_payload,
)

logger = logging.getLogger("auth_picture.routes.audio")
router = APIRouter()


@router.post("/api/audio/enroll")
async def enroll_audio(
    file: UploadFile = File(...),
    user_id: str = Form("anonymous"),
    db: Session = Depends(get_db),
):
    """Enroll an audio file: embed a DWT+QIM watermark with user payload."""
    # Validate extension
    filename = file.filename or "audio.wav"
    if not filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Only WAV files are supported. Install ffmpeg for other formats.",
        )

    # Save uploaded file
    upload_path = UPLOAD_DIR / filename
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Read audio
    try:
        samples, sr = read_wav(upload_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid WAV file: {e}")

    # Generate payload
    payload, hmac_hex = generate_audio_payload(user_id, AUDIO_HMAC_SECRET)

    # Embed watermark
    try:
        watermarked = embed_audio_watermark(
            samples, payload,
            wavelet=AUDIO_DWT_WAVELET,
            level=AUDIO_DWT_LEVEL,
            delta=AUDIO_QIM_DELTA,
            repeat=AUDIO_REPEAT,
        )
    except Exception as e:
        logger.error("Audio watermark embedding failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Watermark embedding failed: {e}")

    # Save output
    output_filename = f"enrolled_{filename}"
    output_path = OUTPUT_DIR / output_filename
    write_wav(output_path, watermarked, sr)

    # Store in DB
    record = AudioWatermark(
        original_filename=filename,
        output_filename=output_filename,
        payload_hex=payload.hex(),
        hmac_hex=hmac_hex,
        sample_rate=sr,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "id": record.id,
        "filename": output_filename,
        "payload_hex": payload.hex(),
        "sample_rate": sr,
        "download_url": f"/api/audio/download/{record.id}",
    }


@router.post("/api/audio/verify")
async def verify_audio(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Verify an audio file: extract watermark and match against DB."""
    filename = file.filename or "audio.wav"
    if not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")

    upload_path = UPLOAD_DIR / f"verify_{filename}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        samples, sr = read_wav(upload_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid WAV file: {e}")

    # Try all known payloads from DB
    all_records = db.query(AudioWatermark).all()

    for record in all_records:
        payload_bytes = bytes.fromhex(record.payload_hex)
        framed_len = len(frame_audio_payload(payload_bytes))

        try:
            extracted = extract_audio_watermark(
                samples, framed_len,
                wavelet=AUDIO_DWT_WAVELET,
                level=AUDIO_DWT_LEVEL,
                delta=AUDIO_QIM_DELTA,
                repeat=AUDIO_REPEAT,
            )
        except Exception:
            continue

        if extracted is not None and extracted == payload_bytes:
            return {
                "verified": True,
                "matched_id": record.id,
                "original_filename": record.original_filename,
                "payload_hex": record.payload_hex,
            }

    return {
        "verified": False,
        "matched_id": None,
        "original_filename": None,
        "payload_hex": None,
    }


@router.get("/api/audio/download/{audio_id}")
async def download_audio(audio_id: int, db: Session = Depends(get_db)):
    """Download an enrolled audio file by its ID."""
    record = db.query(AudioWatermark).filter(AudioWatermark.id == audio_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Audio not found")

    file_path = OUTPUT_DIR / record.output_filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        filename=record.output_filename,
    )
