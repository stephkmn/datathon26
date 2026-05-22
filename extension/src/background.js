import { detectUrl } from "./ml/detector.js";

const CONTEXT_MENU_ID = "check-ai-image";
const LATEST_SCAN_KEY = "latestScan";

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
    const result = await detectUrl(info.srcUrl);

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
