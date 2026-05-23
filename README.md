# [Project Name]

A robust FastAPI-based service designed to detect AI-generated images vs. real content using ONNX-based deep learning models.

## Features
- **Dual Input Methods**: Support for direct file uploads (`/detect/file`) and public URL fetching (`/detect/url`).
- **Safety**: Built-in validation to prevent SSRF (Server-Side Request Forgery) by blocking internal/private IP addresses for URL inputs.
- **Efficient Inference**: Pre-compiled ONNX model support with automatic preprocessing (NCHW format).
- **Robustness**: Includes a stub model for local testing and health checks.

## Setup
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Environment Variables**:
   - `MODEL_PATH`: Path to your `.onnx` file.
   - `USE_STUB_MODEL`: Set to `true` for local development/testing.
   - `ALLOWED_ORIGINS`: Comma-separated list of allowed CORS origins.
3. **Run**: `uvicorn app.main:app --reload`

## Testing
Run the provided test suite to ensure system integrity:
```bash
pytest test_main.py
