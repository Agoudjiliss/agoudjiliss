"""
Auth Picture – Audio Watermark Service
DWT + QIM watermarking for audio files with CRC framing and HMAC verification.
"""

import hmac
import hashlib
import struct
import logging
import wave
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime, timezone

import numpy as np
import pywt

logger = logging.getLogger("auth_picture.audio_watermark")

MAX_AUDIO_PAYLOAD = 32


# ---------------------------------------------------------------------------
# CRC-8 (same as image protection)
# ---------------------------------------------------------------------------
def _crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


# ---------------------------------------------------------------------------
# Payload framing
# ---------------------------------------------------------------------------
def frame_audio_payload(payload: bytes) -> bytes:
    """Frame payload: [length][payload][crc]."""
    if len(payload) > MAX_AUDIO_PAYLOAD:
        payload = payload[:MAX_AUDIO_PAYLOAD]
    frame = bytes([len(payload)]) + payload
    return frame + bytes([_crc8(frame)])


def unframe_audio_payload(frame: bytes) -> Optional[bytes]:
    """Unframe and verify CRC."""
    if len(frame) < 3:
        return None
    length = frame[0]
    if len(frame) < length + 2:
        return None
    payload = frame[1:1 + length]
    crc_expected = frame[1 + length]
    if _crc8(frame[:1 + length]) != crc_expected:
        return None
    return payload


# ---------------------------------------------------------------------------
# HMAC-based payload generation
# ---------------------------------------------------------------------------
def generate_audio_payload(user_id: str, hmac_secret: str) -> Tuple[bytes, str]:
    """Generate a payload and its HMAC for audio watermarking.

    Returns:
        (payload_bytes, hmac_hex)
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    raw = f"{user_id}|{timestamp}".encode("utf-8")[:MAX_AUDIO_PAYLOAD]
    mac = hmac.new(hmac_secret.encode(), raw, hashlib.sha256).hexdigest()
    return raw, mac


# ---------------------------------------------------------------------------
# Audio read/write (WAV)
# ---------------------------------------------------------------------------
def read_wav(path: Path) -> Tuple[np.ndarray, int]:
    """Read a WAV file, return (samples as float64, sample_rate)."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        dtype = np.uint8
    elif sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)

    # If stereo, take first channel
    if n_channels > 1:
        samples = samples[::n_channels]

    return samples, framerate


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write samples as 16-bit WAV file."""
    samples = np.clip(samples, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


# ---------------------------------------------------------------------------
# DWT+QIM embedding for audio
# ---------------------------------------------------------------------------
def embed_audio_watermark(samples: np.ndarray,
                          payload: bytes,
                          wavelet: str = "db4",
                          level: int = 3,
                          delta: float = 50.0,
                          repeat: int = 11) -> np.ndarray:
    """Embed watermark into audio samples using DWT + QIM.

    Args:
        samples: Audio samples as float64.
        payload: Raw payload bytes (will be framed).
        wavelet: PyWavelets wavelet name.
        level: DWT decomposition level.
        delta: QIM quantization step.
        repeat: Repetition factor for robustness.

    Returns:
        Watermarked samples.
    """
    framed = frame_audio_payload(payload)
    bits = np.unpackbits(np.frombuffer(framed, dtype=np.uint8))
    bits = np.repeat(bits, repeat)

    # DWT decompose
    coeffs = pywt.wavedec(samples, wavelet, level=level)

    # Embed in the approximation coefficients
    approx = coeffs[0].copy()
    n = min(len(approx), len(bits))

    for i in range(n):
        quantized = delta * np.floor(approx[i] / delta + 0.5)
        if bits[i] == 1:
            approx[i] = quantized + delta / 4.0
        else:
            approx[i] = quantized - delta / 4.0

    coeffs[0] = approx

    # Reconstruct
    result = pywt.waverec(coeffs, wavelet)
    # Match original length
    return result[:len(samples)]


def extract_audio_watermark(samples: np.ndarray,
                            payload_length: int,
                            wavelet: str = "db4",
                            level: int = 3,
                            delta: float = 50.0,
                            repeat: int = 11) -> Optional[bytes]:
    """Extract watermark from audio samples.

    Args:
        samples: Watermarked audio samples.
        payload_length: Expected framed payload length in bytes.
        wavelet: Must match embedding wavelet.
        level: Must match embedding level.
        delta: Must match embedding delta.
        repeat: Must match embedding repeat.

    Returns:
        Extracted payload bytes or None if CRC fails.
    """
    n_bits = payload_length * 8 * repeat

    coeffs = pywt.wavedec(samples, wavelet, level=level)
    approx = coeffs[0]

    extracted = np.zeros(min(len(approx), n_bits), dtype=np.uint8)
    for i in range(len(extracted)):
        quantized = delta * np.floor(approx[i] / delta + 0.5)
        diff = approx[i] - quantized
        extracted[i] = 1 if diff >= 0 else 0

    # Pad if needed
    if len(extracted) < n_bits:
        extracted = np.concatenate([
            extracted, np.zeros(n_bits - len(extracted), dtype=np.uint8)
        ])
    extracted = extracted[:n_bits]

    # Decode repetition via majority vote
    n = len(extracted) // repeat * repeat
    extracted = extracted[:n]
    reshaped = extracted.reshape(-1, repeat)
    decoded = (np.sum(reshaped, axis=1) > repeat / 2).astype(np.uint8)

    # To bytes
    remainder = len(decoded) % 8
    if remainder:
        decoded = np.concatenate([decoded, np.zeros(8 - remainder, dtype=np.uint8)])
    raw_bytes = np.packbits(decoded).tobytes()

    return unframe_audio_payload(raw_bytes)
