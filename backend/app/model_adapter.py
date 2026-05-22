# backend/app/model_adapter.py

import hashlib
from typing import Dict

from PIL import Image


MODEL_VERSION = "stub-v0"

PLACEHOLDER_EXPLANATION = (
    "Explanation placeholder."
)


def predict_image(image: Image.Image) -> Dict:
    """
    Temporary model stub.

    Later, replace the inside of this function with real model inference.
    Keep the return shape compatible:
      {
        "label": "ai_generated" | "real",
        "confidence": float from 0.0 to 1.0,
        "explanation": str,
        "model_version": str
      }
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize to 32 x 32
    small = image.resize((32, 32))
    payload = small.tobytes() + f"{image.size}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()

    score = digest[0] / 255.0

    p_ai = score

    if p_ai >= 0.5:
        label = "ai-generated"
        confidence = p_ai
    else:
        label = "real"
        confidence = 1.0 - p_ai

    return {
        "label": label,
        "confidence": round(float(confidence), 2),
        "explanation": PLACEHOLDER_EXPLANATION,
        "model_version": MODEL_VERSION,
    }