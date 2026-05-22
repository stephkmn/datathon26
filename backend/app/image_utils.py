# backend/app/image_utils.py

import io
import ipaddress
import os
import socket
from typing import Iterable, NoReturn, TypeAlias
from urllib.parse import urljoin, urlparse

import httpx
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


IPAddress: TypeAlias = ipaddress.IPv4Address | ipaddress.IPv6Address


MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))

SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}

MAX_REDIRECTS = 4
REQUEST_TIMEOUT_SECONDS = 10.0


def api_error(status_code: int, code: str, message: str) -> NoReturn:
    raise HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
        },
    )


def normalize_content_type(content_type: str | None) -> str:
    if not content_type:
        return ""
    return content_type.split(";")[0].strip().lower()


def validate_supported_content_type(content_type: str | None) -> None:
    normalized = normalize_content_type(content_type)

    if normalized not in SUPPORTED_IMAGE_TYPES:
        api_error(
            status_code=415,
            code="unsupported_file_type",
            message="Only JPEG, PNG, WEBP, and GIF images are supported.",
        )


def _resolve_host_ips(hostname: str) -> Iterable[IPAddress]:
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        api_error(
            status_code=400,
            code="invalid_url",
            message="The image URL host could not be resolved.",
        )

    seen: set[str] = set()

    for item in addr_info:
        ip_text = item[4][0]

        if ip_text in seen:
            continue

        seen.add(str(ip_text))

        try:
            ip = ipaddress.ip_address(ip_text)
        except ValueError:
            api_error(
                status_code=400,
                code="invalid_url",
                message="The image URL resolved to an invalid IP address.",
            )

        yield ip


def validate_public_http_url(raw_url: str) -> None:
    parsed = urlparse(raw_url)

    if parsed.scheme not in {"http", "https"}:
        api_error(
            status_code=400,
            code="invalid_url",
            message="Image URL must start with http:// or https://.",
        )

    if not parsed.hostname:
        api_error(
            status_code=400,
            code="invalid_url",
            message="Image URL must include a valid hostname.",
        )

    for ip in _resolve_host_ips(parsed.hostname):
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            api_error(
                status_code=400,
                code="invalid_url",
                message="Image URL must point to a public internet address.",
            )


def bytes_to_rgb_image(data: bytes) -> Image.Image:
    try:
        # First pass verifies the file is a decodable image.
        with Image.open(io.BytesIO(data)) as img:
            img.verify()

        # Reopen after verify(), then convert to RGB for model consistency.
        with Image.open(io.BytesIO(data)) as img:
            return img.convert("RGB")

    except UnidentifiedImageError:
        api_error(
            status_code=415,
            code="unsupported_file_type",
            message="The provided file could not be decoded as an image.",
        )
    except OSError:
        api_error(
            status_code=415,
            code="unsupported_file_type",
            message="The provided image is invalid or corrupted.",
        )


async def fetch_image_from_url(image_url: str):
    current_url = image_url

    timeout = httpx.Timeout(
        REQUEST_TIMEOUT_SECONDS,
        connect=5.0,
    )

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        for _ in range(MAX_REDIRECTS + 1):
            validate_public_http_url(current_url)

            try:
                async with client.stream("GET", current_url) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")

                        if not location:
                            api_error(
                                status_code=400,
                                code="image_fetch_failure",
                                message="Image URL redirected without a Location header.",
                            )

                        current_url = urljoin(current_url, location)
                        continue

                    if response.status_code >= 400:
                        api_error(
                            status_code=400,
                            code="image_fetch_failure",
                            message="Could not fetch image from the provided URL.",
                        )

                    content_type = response.headers.get("content-type")
                    validate_supported_content_type(content_type)

                    content_length = response.headers.get("content-length")
                    if content_length and int(content_length) > MAX_IMAGE_BYTES:
                        api_error(
                            status_code=413,
                            code="file_too_large",
                            message=f"Image is too large. Maximum size is {MAX_IMAGE_BYTES // (1024 * 1024)} MB.",
                        )

                    chunks: list[bytes] = []
                    total = 0

                    async for chunk in response.aiter_bytes():
                        total += len(chunk)

                        if total > MAX_IMAGE_BYTES:
                            api_error(
                                status_code=413,
                                code="file_too_large",
                                message=f"Image is too large. Maximum size is {MAX_IMAGE_BYTES // (1024 * 1024)} MB.",
                            )

                        chunks.append(chunk)

                    return bytes_to_rgb_image(b"".join(chunks))

            except httpx.RequestError:
                api_error(
                    status_code=400,
                    code="image_fetch_failure",
                    message="Could not fetch image from the provided URL.",
                )

    api_error(
        status_code=400,
        code="image_fetch_failure",
        message="Image URL redirected too many times.",
    )


async def load_upload_as_image(upload: UploadFile) -> Image.Image:
    validate_supported_content_type(upload.content_type)

    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = await upload.read(1024 * 1024)

        if not chunk:
            break

        total += len(chunk)

        if total > MAX_IMAGE_BYTES:
            api_error(
                status_code=413,
                code="file_too_large",
                message=f"Image is too large. Maximum size is {MAX_IMAGE_BYTES // (1024 * 1024)} MB.",
            )

        chunks.append(chunk)

    if total == 0:
        api_error(
            status_code=400,
            code="unsupported_file_type",
            message="Uploaded file is empty.",
        )

    return bytes_to_rgb_image(b"".join(chunks))