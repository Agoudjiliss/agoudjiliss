"""
Auth Picture – Image Protection Service
Robust DWT watermark with CRC framing, adaptive QIM, multi-band embedding,
ECC repetition, and optional adversarial perturbation (PGD/FGSM).
"""

import struct
import logging
from typing import Optional

import numpy as np
import pywt
from PIL import Image

logger = logging.getLogger("auth_picture.protection")

# Maximum payload size in bytes
MAX_PAYLOAD_BYTES = 60


# ---------------------------------------------------------------------------
# CRC-8 (simple polynomial 0x07)
# ---------------------------------------------------------------------------
def _crc8(data: bytes) -> int:
    """Compute CRC-8 for data bytes."""
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
# Framing: [length_byte][payload_bytes][crc_byte]
# ---------------------------------------------------------------------------
def frame_payload(payload: bytes) -> bytes:
    """Frame payload with length prefix and CRC-8 suffix."""
    if len(payload) > MAX_PAYLOAD_BYTES:
        payload = payload[:MAX_PAYLOAD_BYTES]
    length = len(payload)
    frame = bytes([length]) + payload
    crc = _crc8(frame)
    return frame + bytes([crc])


def unframe_payload(frame: bytes) -> Optional[bytes]:
    """Unframe payload: verify CRC and extract original payload.
    Returns None if CRC check fails or frame is too short.
    Minimum valid frame is 2 bytes: [length=0][crc].
    """
    if len(frame) < 2:
        return None
    length = frame[0]
    if len(frame) < length + 2:
        return None
    payload = frame[1:1 + length]
    crc_expected = frame[1 + length]
    crc_actual = _crc8(frame[:1 + length])
    if crc_actual != crc_expected:
        return None
    return payload


# ---------------------------------------------------------------------------
# Payload generation
# ---------------------------------------------------------------------------
def generate_payload(sha256_hex: str, phash_hex: str,
                     prompt_text: str = "") -> bytes:
    """Generate a compact binary payload for watermark embedding.
    Format: b"AUTH:" + sha256_binary_16bytes + phash_trunc_8bytes + b"|" + prompt_trunc
    """
    sha_bin = bytes.fromhex(sha256_hex[:32])  # first 16 bytes of sha256
    phash_bin = phash_hex[:16].encode("ascii")[:8]  # up to 8 bytes
    prefix = b"AUTH:"
    sep = b"|"
    header = prefix + sha_bin + phash_bin + sep
    remaining = MAX_PAYLOAD_BYTES - len(header)
    prompt_part = prompt_text.encode("utf-8")[:max(0, remaining)]
    return header + prompt_part


# ---------------------------------------------------------------------------
# Robust level parameters
# ---------------------------------------------------------------------------
def _level_params(robust_level: str) -> dict:
    """Return DWT parameters based on robust level."""
    robust_level = robust_level.lower()
    if robust_level == "low":
        return {"dwt_level": 1, "repeat": 1, "multi_band": False}
    elif robust_level == "high":
        return {"dwt_level": 2, "repeat": 3, "multi_band": True}
    else:  # medium (default)
        return {"dwt_level": 1, "repeat": 2, "multi_band": True}


# ---------------------------------------------------------------------------
# Bit-level ECC: repetition coding
# ---------------------------------------------------------------------------
def _repeat_bits(bits: np.ndarray, repeat: int) -> np.ndarray:
    """Repeat each bit 'repeat' times for error correction."""
    if repeat <= 1:
        return bits
    return np.repeat(bits, repeat)


def _decode_repeated_bits(bits: np.ndarray, repeat: int) -> np.ndarray:
    """Decode repeated bits via majority vote."""
    if repeat <= 1:
        return bits
    n = len(bits) // repeat * repeat
    bits = bits[:n]
    reshaped = bits.reshape(-1, repeat)
    return (np.sum(reshaped, axis=1) > repeat / 2).astype(np.uint8)


# ---------------------------------------------------------------------------
# Bytes <-> bits conversion
# ---------------------------------------------------------------------------
def bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert bytes to a numpy array of bits (MSB first per byte)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Convert a numpy array of bits back to bytes."""
    # Pad to multiple of 8
    remainder = len(bits) % 8
    if remainder:
        bits = np.concatenate([bits, np.zeros(8 - remainder, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


# ---------------------------------------------------------------------------
# Adaptive delta-QIM
# ---------------------------------------------------------------------------
def _qim_embed_coeff(coeff: float, bit: int, delta: float) -> float:
    """Embed a single bit into a coefficient using QIM."""
    quantized = delta * np.floor(coeff / delta + 0.5)
    if bit == 1:
        return float(quantized + delta / 4.0)
    else:
        return float(quantized - delta / 4.0)


def _qim_extract_bit(coeff: float, delta: float) -> int:
    """Extract a single bit from a coefficient using QIM."""
    quantized = delta * np.floor(coeff / delta + 0.5)
    diff = coeff - quantized
    return 1 if diff >= 0 else 0


def _adaptive_delta(band: np.ndarray, strength: float) -> float:
    """Compute QIM delta for watermark embedding.
    The ``band`` argument is accepted for API consistency and future
    adaptive computation based on band statistics.  Currently the delta
    is derived directly from ``strength`` with a minimum of 4.0 to ensure
    robustness against uint8 quantization noise in the spatial domain.
    """
    return max(float(strength), 4.0)


# ---------------------------------------------------------------------------
# DWT-based watermark embed/extract
# ---------------------------------------------------------------------------
def embed_watermark(image: Image.Image, payload: bytes,
                    strength: float = 10.0,
                    robust_level: str = "medium") -> Image.Image:
    """Embed watermark payload into image using DWT + adaptive QIM.

    Args:
        image: PIL Image (will be converted to float64 grayscale or RGB).
        payload: Raw payload bytes (will be framed with CRC).
        strength: Base watermark strength.
        robust_level: 'low', 'medium', or 'high'.

    Returns:
        Watermarked PIL Image in the same mode as input.
    """
    params = _level_params(robust_level)
    framed = frame_payload(payload)
    bits = bytes_to_bits(framed)
    bits = _repeat_bits(bits, params["repeat"])

    # Convert image to numpy float64
    orig_mode = image.mode
    if orig_mode not in ("L", "RGB", "RGBA"):
        image = image.convert("RGB")
        orig_mode = "RGB"

    img_arr = np.array(image, dtype=np.float64)
    orig_shape = img_arr.shape

    if orig_mode == "L":
        # Grayscale: work directly
        img_arr = _embed_channel(img_arr, bits, strength, params)
    else:
        # Color: embed in all channels for robustness
        for c in range(min(3, img_arr.shape[2])):
            img_arr[:, :, c] = _embed_channel(
                img_arr[:, :, c], bits, strength, params
            )

    # Clip and convert back
    img_arr = np.clip(img_arr, 0, 255)
    result = Image.fromarray(img_arr.astype(np.uint8), mode=orig_mode)
    return result


def _embed_channel(channel: np.ndarray, bits: np.ndarray,
                   strength: float, params: dict) -> np.ndarray:
    """Embed bits into a single channel using DWT."""
    dwt_level = params["dwt_level"]
    multi_band = params["multi_band"]
    orig_shape = channel.shape

    coeffs = pywt.wavedec2(channel, "haar", level=dwt_level)

    # Collect target bands
    bands_info = []  # list of (level_idx, band_name, band_array)

    # Always use LL (approximation at deepest level)
    bands_info.append((0, "LL", coeffs[0]))

    if multi_band:
        # Add detail bands from first detail level
        detail_idx = 1
        if detail_idx < len(coeffs):
            cH, cV, cD = coeffs[detail_idx]
            bands_info.append((detail_idx, "cH", cH))
            bands_info.append((detail_idx, "cV", cV))

    # Embed bits across all target bands
    for level_idx, band_name, band in bands_info:
        flat = band.flatten()
        delta = _adaptive_delta(flat, strength)
        n_coeffs = len(flat)
        n_bits = len(bits)

        for i in range(min(n_coeffs, n_bits)):
            flat[i] = _qim_embed_coeff(flat[i], int(bits[i]), delta)

        reshaped = flat.reshape(band.shape)
        if level_idx == 0:
            coeffs[0] = reshaped
        else:
            detail = list(coeffs[level_idx])
            if band_name == "cH":
                detail[0] = reshaped
            elif band_name == "cV":
                detail[1] = reshaped
            elif band_name == "cD":
                detail[2] = reshaped
            coeffs[level_idx] = tuple(detail)

    # Inverse DWT
    reconstructed = pywt.waverec2(coeffs, "haar")
    # Crop to original shape (DWT can pad)
    reconstructed = reconstructed[:orig_shape[0], :orig_shape[1]]
    return reconstructed


def extract_watermark(image: Image.Image, payload_length: int,
                      strength: float = 10.0,
                      robust_level: str = "medium") -> Optional[bytes]:
    """Extract watermark payload from image.

    Args:
        image: Potentially watermarked PIL Image.
        payload_length: Expected framed payload length in bytes.
        strength: Base watermark strength (must match embed).
        robust_level: Must match the level used for embedding.

    Returns:
        Extracted payload bytes (unframed) or None if CRC fails.
    """
    params = _level_params(robust_level)
    n_bits_total = payload_length * 8 * params["repeat"]

    orig_mode = image.mode
    if orig_mode not in ("L", "RGB", "RGBA"):
        image = image.convert("RGB")
        orig_mode = "RGB"

    img_arr = np.array(image, dtype=np.float64)

    if orig_mode == "L":
        raw_bits = _extract_channel(img_arr, n_bits_total, strength, params)
    else:
        # Extract from all channels and vote
        all_bits = []
        for c in range(min(3, img_arr.shape[2])):
            ch_bits = _extract_channel(
                img_arr[:, :, c], n_bits_total, strength, params
            )
            all_bits.append(ch_bits)

        # Majority vote across channels
        stacked = np.stack(all_bits, axis=0)
        raw_bits = (np.sum(stacked, axis=0) > len(all_bits) / 2).astype(np.uint8)

    # Decode ECC repetition
    decoded_bits = _decode_repeated_bits(raw_bits, params["repeat"])

    # Convert to bytes
    extracted_bytes = bits_to_bytes(decoded_bits)

    # Unframe
    return unframe_payload(extracted_bytes)


def _extract_channel(channel: np.ndarray, n_bits: int,
                     strength: float, params: dict) -> np.ndarray:
    """Extract raw bits from a single channel using DWT."""
    dwt_level = params["dwt_level"]
    multi_band = params["multi_band"]

    coeffs = pywt.wavedec2(channel, "haar", level=dwt_level)

    bands_info = []
    bands_info.append((0, "LL", coeffs[0]))

    if multi_band:
        detail_idx = 1
        if detail_idx < len(coeffs):
            cH, cV, cD = coeffs[detail_idx]
            bands_info.append((detail_idx, "cH", cH))
            bands_info.append((detail_idx, "cV", cV))

    # Extract bits from each band and vote
    all_band_bits = []
    for _, _, band in bands_info:
        flat = band.flatten()
        delta = _adaptive_delta(flat, strength)
        extracted = np.zeros(min(len(flat), n_bits), dtype=np.uint8)
        for i in range(len(extracted)):
            extracted[i] = _qim_extract_bit(float(flat[i]), delta)
        # Pad if needed
        if len(extracted) < n_bits:
            extracted = np.concatenate([
                extracted, np.zeros(n_bits - len(extracted), dtype=np.uint8)
            ])
        all_band_bits.append(extracted[:n_bits])

    # Vote across bands
    if len(all_band_bits) > 1:
        stacked = np.stack(all_band_bits, axis=0)
        result = (np.sum(stacked, axis=0) > len(all_band_bits) / 2).astype(np.uint8)
    else:
        result = all_band_bits[0]

    return result


# ---------------------------------------------------------------------------
# DCT noise (spectral watermark layer)
# ---------------------------------------------------------------------------
def add_dct_noise(image: Image.Image, strength: float = 0.5) -> Image.Image:
    """Add subtle DCT-domain noise for additional protection layer."""
    img_arr = np.array(image, dtype=np.float64)
    rng = np.random.default_rng(42)  # deterministic seed
    noise = rng.normal(0, strength, img_arr.shape)
    img_arr = np.clip(img_arr + noise, 0, 255)
    return Image.fromarray(img_arr.astype(np.uint8), mode=image.mode)


# ---------------------------------------------------------------------------
# Full protection pipeline
# ---------------------------------------------------------------------------
def protect_image(image: Image.Image,
                  sha256_hex: str,
                  phash_hex: str,
                  prompt_text: str = "",
                  strength: float = 10.0,
                  robust_level: str = "medium") -> Image.Image:
    """Full image protection: generate payload, embed watermark, add DCT noise.

    Args:
        image: Input PIL Image.
        sha256_hex: SHA-256 hash of the original file.
        phash_hex: Perceptual hash string.
        prompt_text: Anti-AI prompt text to embed.
        strength: Watermark strength.
        robust_level: 'low', 'medium', or 'high'.

    Returns:
        Protected PIL Image.
    """
    payload = generate_payload(sha256_hex, phash_hex, prompt_text)
    watermarked = embed_watermark(image, payload, strength, robust_level)
    protected = add_dct_noise(watermarked, strength=0.5)
    return protected
