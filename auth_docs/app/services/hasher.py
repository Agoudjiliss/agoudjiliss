"""
Auth Picture – Hashing Service
SHA-256 and perceptual hash (pHash) for images.
"""

import hashlib
from pathlib import Path

import imagehash
from PIL import Image


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest for raw bytes."""
    return hashlib.sha256(data).hexdigest()


def phash_image(path: Path) -> str:
    """Compute perceptual hash (pHash) of an image file."""
    img = Image.open(path)
    return str(imagehash.phash(img))


def phash_distance(h1: str, h2: str) -> int:
    """Hamming distance between two pHash hex strings."""
    a = imagehash.hex_to_hash(h1)
    b = imagehash.hex_to_hash(h2)
    return int(a - b)
