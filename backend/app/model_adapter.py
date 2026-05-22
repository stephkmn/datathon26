import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from dotenv import load_dotenv
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(REPO_ROOT / ".env")

AI_THRESHOLD = 0.33
REAL_THRESHOLD = 0.67

LabelValue = str

PLACEHOLDER_EXPLANATION = (
    "Explanation placeholder."
)

# Verify model path
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

# --- LOAD MODEL ---
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

# Helper function to convert probability to confidence
def _derive_confidence_from_prob(prob):
    if prob <= AI_THRESHOLD:
        return "ai-generated", (prob / AI_THRESHOLD)
    elif prob >= REAL_THRESHOLD:
        return "real", ((prob - REAL_THRESHOLD) / AI_THRESHOLD)
    else:
        return "unknown", ((prob - AI_THRESHOLD) / AI_THRESHOLD)

# --- INFERENCE ---
def predict_image(image: Image.Image) -> dict[str, Any]:
    if USE_STUB_MODEL:
        prob =  _predict_stub(image)
    else:
        prob = _predict_onnx(image)

    label, confidence = _derive_confidence_from_prob(prob)

    return {
        "label": label,
        "confidence": round(float(confidence), 2),
        "explanation": PLACEHOLDER_EXPLANATION,
        "model_version": MODEL_VERSION,
    }

def _predict_stub(image: Image.Image) -> float:
    if image.mode != "RGB":
        image = image.convert("RGB")

    small = image.resize((32, 32))
    payload = small.tobytes() + f"{image.size}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()

    p_ai = digest[0] / 255.0
    return p_ai

def evaluate(image: Image.Image) -> float:
    """Returns confidence value 0 to 1 (0=FAKE, 1=REAL)."""
    if _onnx_session is None:
        raise ValueError("Onnx session is none.")
    if _onnx_input_name is None:
        raise ValueError("ONNX input name is unknown.")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(image.convert("RGB")).unsqueeze(0)
    img_numpy = img_tensor.numpy().astype(np.float32, copy=False)
    outputs = _onnx_session.run(None, {_onnx_input_name: img_numpy})

    confidence = float(outputs[0].flatten()[0])
    return max(0.0, min(1.0, confidence))

def _predict_onnx(image: Image.Image) -> float:
    if _onnx_session is None or _onnx_input_name is None:
        message = _load_error or "ONNX model is not loaded."
        raise ModelAdapterError(message)

    p_ai = evaluate(image)
    return p_ai

_load_onnx_model()
