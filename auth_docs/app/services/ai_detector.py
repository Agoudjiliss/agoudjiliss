"""
Auth Picture – AI Detection Service
Detects AI-generated images using Hugging Face ViT-based classifiers.
Models are loaded lazily on first use and cached.
"""

import logging
from typing import Optional, Tuple

from PIL import Image

logger = logging.getLogger("auth_picture.ai_detector")

# Lazy-loaded model cache
_pipelines: dict = {}


def _get_pipeline(model_name: str):
    """Lazily load a Hugging Face image-classification pipeline."""
    if model_name not in _pipelines:
        try:
            from transformers import pipeline as hf_pipeline
            from auth_docs.config import HF_CACHE_DIR
            logger.info("Loading AI detection model: %s", model_name)
            _pipelines[model_name] = hf_pipeline(
                "image-classification",
                model=model_name,
                cache_dir=str(HF_CACHE_DIR),
            )
        except Exception as e:
            logger.error("Failed to load model %s: %s", model_name, e)
            return None
    return _pipelines[model_name]


def detect_ai(image: Image.Image,
              model_1: Optional[str] = None,
              model_2: Optional[str] = None,
              threshold: Optional[float] = None) -> dict:
    """Run AI detection on an image with two models.

    Returns:
        dict with keys: score_1, score_2, label_1, label_2, is_ai, avg_score
    """
    from auth_docs.config import AI_MODEL_1, AI_MODEL_2, AI_THRESHOLD

    m1 = model_1 or AI_MODEL_1
    m2 = model_2 or AI_MODEL_2
    thr = threshold if threshold is not None else AI_THRESHOLD

    result: dict = {
        "score_1": None, "score_2": None,
        "label_1": None, "label_2": None,
        "is_ai": False, "avg_score": 0.0,
    }

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    for idx, model_name in enumerate([m1, m2], start=1):
        pipe = _get_pipeline(model_name)
        if pipe is None:
            logger.warning("Model %s unavailable, skipping.", model_name)
            continue
        try:
            predictions = pipe(image)
            # predictions is a list of dicts: [{"label": ..., "score": ...}, ...]
            ai_score = 0.0
            ai_label = "human"
            for pred in predictions:
                label = pred["label"].lower()
                if "ai" in label or "artificial" in label or "fake" in label:
                    ai_score = pred["score"]
                    ai_label = pred["label"]
                    break
            result[f"score_{idx}"] = ai_score
            result[f"label_{idx}"] = ai_label
        except Exception as e:
            logger.error("Error running model %s: %s", model_name, e)

    scores = [s for s in [result["score_1"], result["score_2"]] if s is not None]
    if scores:
        result["avg_score"] = sum(scores) / len(scores)
        result["is_ai"] = result["avg_score"] >= thr

    return result
