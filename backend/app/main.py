# backend/app/main.py

import os

from schemas import InputType, Label

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.image_utils import api_error, fetch_image_from_url, load_upload_as_image
from app.model_adapter import MODEL_VERSION, PLACEHOLDER_EXPLANATION, predict_image
from app.schemas import DetectionResponse, HealthResponse, ImageUrlRequest


ALLOWED_LABELS = {"ai_generated", "real"}


def parse_allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(
    title="AI Image Detector API",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=MODEL_VERSION,
    )


def build_detection_response(prediction: dict, input_type: InputType) -> DetectionResponse:
    label = prediction.get("label", "unknown")

    if label not in ALLOWED_LABELS:
        raise ValueError("Invalid label.")

    try:
        confidence = float(prediction.get("confidence", 0.0))
    except TypeError:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    explanation = prediction.get("explanation") or PLACEHOLDER_EXPLANATION
    model_version = prediction.get("model_version") or MODEL_VERSION

    return DetectionResponse(
        label=label,
        confidence=confidence,
        explanation=explanation,
        model_version=model_version,
        input_type=input_type,
    )


@app.post("/detect/url", response_model=DetectionResponse)
async def detect_url(payload: ImageUrlRequest) -> DetectionResponse:
    image = await fetch_image_from_url(payload.image_url)

    try:
        prediction = predict_image(image)
    except Exception:
        api_error(
            status_code=500,
            code="model_inference_failure",
            message="Model inference failed.",
        )

    return build_detection_response(prediction, input_type="url")


@app.post("/detect/file", response_model=DetectionResponse)
async def detect_file(image: UploadFile = File(...)) -> DetectionResponse:
    pil_image = await load_upload_as_image(image)

    try:
        prediction = predict_image(pil_image)
    except Exception:
        api_error(
            status_code=500,
            code="model_inference_failure",
            message="Model inference failed.",
        )

    return build_detection_response(prediction, input_type="file")