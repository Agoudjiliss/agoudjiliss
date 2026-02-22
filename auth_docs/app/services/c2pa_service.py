"""
Auth Picture – C2PA Service
Sign and verify images using the C2PA standard (Content Credentials).
Uses the c2pa-python library when available, with graceful fallback.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("auth_picture.c2pa_service")

# Try importing c2pa
_c2pa_available = False
try:
    import c2pa  # type: ignore
    _c2pa_available = True
except ImportError:
    logger.warning("c2pa-python not installed. C2PA signing/verification disabled.")


def sign_image(input_path: Path, output_path: Path,
               cert_path: Path, key_path: Path,
               claim_generator: str = "AuthPicture/0.2.0",
               prompt_text: str = "") -> bool:
    """Sign an image file with C2PA manifest.

    Args:
        input_path: Path to the image to sign.
        output_path: Path for the signed output.
        cert_path: Path to PEM certificate.
        key_path: Path to PEM private key.
        claim_generator: Generator string for the manifest.
        prompt_text: Anti-AI prompt text to include as custom assertion.

    Returns:
        True if signing succeeded, False otherwise.
    """
    if not _c2pa_available:
        logger.warning("C2PA not available, skipping signing.")
        return False

    try:
        cert_bytes = cert_path.read_bytes()
        key_bytes = key_path.read_bytes()

        # Build manifest JSON
        assertions = [
            {
                "label": "c2pa.training-mining",
                "data": {
                    "entries": {
                        "c2pa.ai_generative_training": {"use": "notAllowed"},
                        "c2pa.ai_inference": {"use": "notAllowed"},
                        "c2pa.ai_training": {"use": "notAllowed"},
                        "c2pa.data_mining": {"use": "notAllowed"},
                    }
                },
            },
        ]

        # Add custom prompt assertion if provided
        if prompt_text:
            assertions.append({
                "label": "auth_picture.prompt",
                "data": {"text": prompt_text},
            })

        manifest_def = {
            "claim_generator": claim_generator,
            "assertions": assertions,
        }

        manifest_json = json.dumps(manifest_def)

        # Use c2pa Builder API
        builder = c2pa.Builder(manifest_json)
        # Sign the file
        builder.sign_file(
            str(input_path),
            str(output_path),
            c2pa.create_signer(cert_bytes, key_bytes, c2pa.SigningAlg.ES256),
        )

        logger.info("C2PA signed: %s -> %s", input_path, output_path)
        return True

    except Exception as e:
        logger.error("C2PA signing failed: %s", e)
        return False


def verify_image(image_path: Path) -> dict:
    """Verify C2PA manifest on an image.

    Returns:
        dict with keys: valid (bool), manifest (dict or None), error (str or None)
    """
    if not _c2pa_available:
        return {"valid": False, "manifest": None, "error": "c2pa-python not installed"}

    try:
        reader = c2pa.Reader.from_file(str(image_path))
        manifest_store = json.loads(reader.json())

        # Check if there's an active manifest
        active = manifest_store.get("active_manifest")
        if active:
            return {"valid": True, "manifest": manifest_store, "error": None}
        else:
            return {"valid": False, "manifest": manifest_store, "error": "No active manifest"}

    except Exception as e:
        logger.error("C2PA verification failed: %s", e)
        return {"valid": False, "manifest": None, "error": str(e)}
