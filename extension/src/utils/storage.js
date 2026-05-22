export const LATEST_SCAN_KEY = "latestScan";

export const EMPTY_SCAN_STATE = {
  status: "idle",
  result: null,
  error: "",
  source: "",
};

export function getLatestScan() {
  return chrome.storage.local
    .get(LATEST_SCAN_KEY)
    .then((items) => items[LATEST_SCAN_KEY] || EMPTY_SCAN_STATE);
}

export function setLatestScan(state) {
  return chrome.storage.local.set({
    [LATEST_SCAN_KEY]: {
      ...EMPTY_SCAN_STATE,
      ...state,
    },
  });
}
