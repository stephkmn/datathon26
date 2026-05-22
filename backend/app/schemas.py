# backend/app/schemas.py

from typing import Literal

from pydantic import BaseModel, Field


Label = Literal["ai-generated", "real"]
InputType = Literal["url", "file"]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class ImageUrlRequest(BaseModel):
    image_url: str = Field(..., min_length=1)


class DetectionResponse(BaseModel):
    label: Label
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    model_version: str
    input_type: InputType
