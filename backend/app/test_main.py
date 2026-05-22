import importlib
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main, model_adapter


client = TestClient(main.app)


def make_image_bytes(
    image_format: str = "PNG",
    size: tuple[int, int] = (16, 16),
    color: tuple[int, int, int] = (255, 255, 255),
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=image_format)
    return buffer.getvalue()


def test_health_reports_stub_model_loaded():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": "stub-v0",
    }


def test_detect_file_accepts_supported_image_and_returns_contract():
    image_bytes = make_image_bytes()

    response = client.post(
        "/detect/file",
        files={"image": ("test.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200

    body = response.json()
    assert body["label"] in {"ai-generated", "real"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["explanation"]
    assert body["model_version"] == "stub-v0"
    assert body["input_type"] == "file"


def test_detect_file_rejects_unsupported_content_type():
    response = client.post(
        "/detect/file",
        files={"image": ("test.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "unsupported_file_type"


def test_detect_file_rejects_corrupt_image_even_with_supported_content_type():
    response = client.post(
        "/detect/file",
        files={"image": ("broken.png", b"not actually a png", "image/png")},
    )

    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "unsupported_file_type"


def test_detect_file_rejects_large_upload(monkeypatch):
    monkeypatch.setattr("app.image_utils.MAX_IMAGE_BYTES", 10)

    response = client.post(
        "/detect/file",
        files={"image": ("test.png", make_image_bytes(), "image/png")},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "file_too_large"


def test_detect_url_rejects_non_http_url():
    response = client.post(
        "/detect/url",
        json={"image_url": "file:///tmp/image.png"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_url"


def test_detect_url_rejects_localhost_url():
    response = client.post(
        "/detect/url",
        json={"image_url": "http://localhost:8000/image.png"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_url"


def test_build_detection_response_rejects_unsupported_model_label():
    with pytest.raises(Exception) as exc_info:
        main.build_detection_response(
            {
                "label": "unknown",
                "confidence": 0.9,
                "explanation": "bad label",
                "model_version": "test",
            },
            input_type="file",
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail["code"] == "model_inference_failure"


def test_build_detection_response_rejects_invalid_confidence():
    with pytest.raises(Exception) as exc_info:
        main.build_detection_response(
            {
                "label": "real",
                "confidence": "not-a-number",
                "explanation": "bad confidence",
                "model_version": "test",
            },
            input_type="file",
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail["code"] == "model_inference_failure"


def test_stub_prediction_is_deterministic_and_uses_allowed_labels():
    image = Image.new("RGB", (12, 12), (20, 40, 60))

    first = model_adapter.predict_image(image)
    second = model_adapter.predict_image(image)

    assert first == second
    assert first["label"] in {"ai-generated", "real"}
    assert 0.0 <= first["confidence"] <= 1.0


def test_preprocess_image_uses_pytorch_channel_first_shape():
    image = Image.new("RGB", (64, 48), (255, 0, 0))

    tensor = model_adapter._preprocess_image(image)

    assert tensor.shape == (1, 3, 32, 32)
    assert tensor.dtype == np.float32
    assert tensor.flags["C_CONTIGUOUS"]
    assert np.allclose(tensor[0, 0], 1.0)
    assert np.allclose(tensor[0, 1], 0.0)
    assert np.allclose(tensor[0, 2], 0.0)


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        ([np.array([0.8], dtype=np.float32)], 0.8),
        ([np.array([0.2, 0.8], dtype=np.float32)], 0.8),
        ([np.array([0.0, 2.0], dtype=np.float32)], 0.8807970779778823),
    ],
)
def test_probability_from_outputs_handles_common_model_outputs(outputs, expected):
    p_ai = model_adapter._probability_from_outputs(outputs)

    assert p_ai == pytest.approx(expected)


def test_onnx_mode_without_model_path_does_not_crash(monkeypatch):
    monkeypatch.setenv("USE_STUB_MODEL", "false")
    monkeypatch.delenv("MODEL_PATH", raising=False)

    reloaded = importlib.reload(model_adapter)

    try:
        assert reloaded.is_model_loaded() is False
        assert "MODEL_PATH is required" in reloaded.get_model_load_error()
    finally:
        monkeypatch.delenv("USE_STUB_MODEL", raising=False)
        importlib.reload(model_adapter)
