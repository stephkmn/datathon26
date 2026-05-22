const CONTEXT_MENU_ID = "check-ai-image";
const LATEST_SCAN_KEY = "latestScan";
const BACKEND_URL = "http://localhost:8000";

const emptyScanState = {
  status: "idle",
  result: null,
  error: "",
  source: "",
};

async function setLatestScan(state) {
  await chrome.storage.local.set({
    [LATEST_SCAN_KEY]: {
      ...emptyScanState,
      ...state,
    },
  });
}

async function detectImageUrl(imageUrl) {
  const response = await fetch(`${BACKEND_URL}/detect/url`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      image_url: imageUrl,
    }),
  });

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    const message =
      data?.detail?.message || data?.message || "Backend request failed.";
    throw new Error(message);
  }

  return data;
}

function createContextMenu() {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: CONTEXT_MENU_ID,
      title: "Check if image is AI-generated",
      contexts: ["image"],
    });
  });
}

chrome.runtime.onInstalled.addListener(createContextMenu);
chrome.runtime.onStartup.addListener(createContextMenu);

chrome.contextMenus.onClicked.addListener(async (info) => {
  if (info.menuItemId !== CONTEXT_MENU_ID || !info.srcUrl) {
    return;
  }

  await setLatestScan({
    status: "loading",
    source: "right-clicked URL",
  });

  try {
    const result = await detectImageUrl(info.srcUrl);

    await setLatestScan({
      status: "success",
      result,
      source: "right-clicked URL",
    });

    await chrome.action.setBadgeText({ text: "OK" });
    await chrome.action.setBadgeBackgroundColor({ color: "#2563eb" });

    if (chrome.action.openPopup) {
      await chrome.action.openPopup().catch(() => {});
    }
  } catch (error) {
    await setLatestScan({
      status: "error",
      error: error.message || "Could not check this image.",
      source: "right-clicked URL",
    });

    await chrome.action.setBadgeText({ text: "!" });
    await chrome.action.setBadgeBackgroundColor({ color: "#dc2626" });
  }
});
