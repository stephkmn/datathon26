# backend/app/model_adapter.py

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(REPO_ROOT / ".env")

AI_THRESHOLD = 0.33
REAL_THRESHOLD = 0.67

LabelValue = str

PLACEHOLDER_EXPLANATION = (
    "Explanation placeholder."
)

def _resolve_model_path(model_path: str) -> str:
    if not model_path.strip():
        return ""

    raw_path = Path(model_path).expanduser()

    if raw_path.is_absolute():
        return str(raw_path.resolve())

    candidates = [
        Path.cwd() / raw_path,
        REPO_ROOT / raw_path,
        BACKEND_ROOT / raw_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return str((REPO_ROOT / raw_path).resolve())


MODEL_PATH = _resolve_model_path(os.getenv("MODEL_PATH", ""))
MODEL_VERSION = os.getenv("MODEL_VERSION", "stub-v0").strip() or "stub-v0"
try:
    MODEL_INPUT_SIZE = int(os.getenv("MODEL_INPUT_SIZE", "32"))
except ValueError:
    MODEL_INPUT_SIZE = 32
MODEL_RESAMPLE = os.getenv("MODEL_RESAMPLE", "bilinear").strip().lower()
MODEL_RESIZE_MODE = os.getenv("MODEL_RESIZE_MODE", "stretch").strip().lower()

RESAMPLE_FILTERS = {
    "nearest": Image.Resampling.NEAREST,
    "box": Image.Resampling.BOX,
    "bilinear": Image.Resampling.BILINEAR,
    "hamming": Image.Resampling.HAMMING,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}
RESAMPLE_FILTER = RESAMPLE_FILTERS.get(MODEL_RESAMPLE, Image.Resampling.BILINEAR)


def _env_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    return None


_USE_STUB_OVERRIDE = _env_bool("USE_STUB_MODEL")
USE_STUB_MODEL = _USE_STUB_OVERRIDE if _USE_STUB_OVERRIDE is not None else not MODEL_PATH

_onnx_session: Any | None = None
_onnx_input_name: str | None = None
_onnx_input_rank: int | None = None
_load_error: str | None = None


class ModelAdapterError(RuntimeError):
    """Raised when the configured model cannot produce a valid prediction."""


def _load_onnx_model() -> None:
    global _onnx_input_name, _onnx_input_rank, _onnx_session, _load_error

    if USE_STUB_MODEL:
        return

    if not MODEL_PATH:
        _load_error = "MODEL_PATH is required when USE_STUB_MODEL=false."
        return

    path = Path(MODEL_PATH)
    if not path.exists():
        _load_error = f"Model file does not exist: {MODEL_PATH}"
        return

    try:
        import onnxruntime as ort

        _onnx_session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        model_input = _onnx_session.get_inputs()[0]
        _onnx_input_name = model_input.name
        _onnx_input_rank = len(model_input.shape)
        _load_error = None
    except Exception as exc:
        _onnx_session = None
        _onnx_input_name = None
        _onnx_input_rank = None
        _load_error = f"Failed to load ONNX model: {exc}"


def is_model_loaded() -> bool:
    return USE_STUB_MODEL or _onnx_session is not None


def get_model_load_error() -> str | None:
    return _load_error


def _predict_stub(image: Image.Image) -> dict[str, Any]:
    if image.mode != "RGB":
        image = image.convert("RGB")

    small = image.resize((32, 32))
    payload = small.tobytes() + f"{image.size}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()

    p_ai = digest[0] / 255.0
    label, confidence = derive_confidence_from_prob(p_ai)

    return {
        "label": label,
        "confidence": round(float(confidence), 2),
        "explanation": PLACEHOLDER_EXPLANATION,
        "model_version": MODEL_VERSION,
    }


def _resize_for_model(image: Image.Image) -> Image.Image:
    size = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

    if MODEL_RESIZE_MODE == "center_crop":
        return ImageOps.fit(
            image,
            size,
            method=RESAMPLE_FILTER,
            centering=(0.5, 0.5),
        )

    if MODEL_RESIZE_MODE == "letterbox":
        contained = ImageOps.contain(image, size, method=RESAMPLE_FILTER)
        canvas = Image.new("RGB", size, (0, 0, 0))
        offset = (
            (MODEL_INPUT_SIZE - contained.width) // 2,
            (MODEL_INPUT_SIZE - contained.height) // 2,
        )
        canvas.paste(contained, offset)
        return canvas

    return image.resize(size, resample=RESAMPLE_FILTER)


def _preprocess_image(image: Image.Image) -> np.ndarray:
    """
    ONNX preprocessing for a 32 x 32 RGB model.

    The model contract follows the PyTorch image convention: C,H,W.
    PIL/numpy images arrive as H,W,C, so channels are moved first.
    """
    rgb = _resize_for_model(image.convert("RGB"))
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    chw = np.ascontiguousarray(chw, dtype=np.float32)

    if _onnx_input_rank == 3:
        return chw

    return np.expand_dims(chw, axis=0)


def _probability_from_outputs(outputs: list[np.ndarray]) -> float:
    if not outputs:
        raise ModelAdapterError("ONNX model returned no outputs.")

    output = np.asarray(outputs[0]).squeeze()

    if output.size == 1:
        value = float(output.item())
        if 0.0 <= value <= 1.0:
            return value

        return float(1.0 / (1.0 + np.exp(-value)))

    if output.size >= 2:
        flat = output.reshape(-1).astype(np.float64)

        if np.all((0.0 <= flat) & (flat <= 1.0)) and np.isclose(np.sum(flat), 1.0):
            return float(flat[1])

        shifted = flat - np.max(flat)
        probs = np.exp(shifted) / np.sum(np.exp(shifted))
        return float(probs[1])

    raise ModelAdapterError("ONNX model output shape is unsupported.")


def _predict_onnx(image: Image.Image) -> dict[str, Any]:
    if _onnx_session is None or _onnx_input_name is None:
        message = _load_error or "ONNX model is not loaded."
        raise ModelAdapterError(message)

    tensor = _preprocess_image(image)
    outputs = _onnx_session.run(None, {_onnx_input_name: tensor})
    p_ai = _probability_from_outputs(outputs)
    p_ai = max(0.0, min(1.0, float(p_ai)))

    label, confidence = derive_confidence_from_prob(p_ai)

    return {
        "label": label,
        "confidence": float(confidence),
        "explanation": PLACEHOLDER_EXPLANATION,
        "model_version": MODEL_VERSION,
    }


def predict_image(image: Image.Image) -> dict[str, Any]:
    if USE_STUB_MODEL:
        return _predict_stub(image)

    return _predict_onnx(image)


def derive_confidence_from_prob(prob):
    if prob <= AI_THRESHOLD:
        return "ai-generated", (prob / AI_THRESHOLD)
    elif prob >= REAL_THRESHOLD:
        return "real", ((prob - REAL_THRESHOLD) / AI_THRESHOLD)
    else:
        return "unknown", ((prob - AI_THRESHOLD) / AI_THRESHOLD)



_load_onnx_model()
