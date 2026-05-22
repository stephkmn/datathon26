const DEFAULT_BACKEND_URL = "http://localhost:8000";

export function getBackendUrl() {
  return DEFAULT_BACKEND_URL;
}

async function parseResponse(response) {
  const data = await response.json().catch(() => null);

  if (!response.ok) {
    const message =
      data?.detail?.message || data?.message || "Backend request failed.";
    throw new Error(message);
  }

  return data;
}

export async function detectFile(file) {
  const formData = new FormData();
  formData.append("image", file);

  const response = await fetch(`${getBackendUrl()}/detect/file`, {
    method: "POST",
    body: formData,
  });

  return parseResponse(response);
}

export async function detectUrl(imageUrl) {
  const response = await fetch(`${getBackendUrl()}/detect/url`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      image_url: imageUrl,
    }),
  });

  return parseResponse(response);
}
