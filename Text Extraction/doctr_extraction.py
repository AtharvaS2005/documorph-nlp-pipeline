"""
DocTR OCR fallback engine.

Used as a second-opinion fallback when PaddleOCR fails or returns
insufficient text, even after its own internal fallback/rescue passes.
General-purpose -- not tuned to any specific document layout or domain.
"""

from typing import Tuple

import numpy as np

_MODEL_CACHE = {}


def _get_model():
    model = _MODEL_CACHE.get("model")
    if model is not None:
        return model

    from doctr.models import ocr_predictor

    model = ocr_predictor(pretrained=True)
    _MODEL_CACHE["model"] = model
    return model


def extract_text(image_bgr: np.ndarray) -> Tuple[bool, str, str]:
    """Run DocTR on a single BGR image and return (ok, text, err), matching
    the same (ok, text, err) interface as PaddleEngine.extract_text."""
    if image_bgr is None:
        return False, "", "no image provided"

    try:
        import cv2

        model = _get_model()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = model([rgb])
        text = result.render().strip()
        if text:
            return True, text, ""
        return False, "", "DocTR returned no text"
    except Exception as exc:
        return False, "", str(exc)