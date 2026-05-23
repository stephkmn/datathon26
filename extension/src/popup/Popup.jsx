import { useEffect, useState } from "react";

import { detectFile } from "../api/client.js";
import {
  EMPTY_SCAN_STATE,
  LATEST_SCAN_KEY,
  getLatestScan,
  setLatestScan,
} from "../utils/storage.js";

function formatConfidence(value) {
  if (typeof value !== "number") {
    return "";
  }

  return `${Math.round(value * 100)}%`;
}

export default function Popup() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [scanState, setScanState] = useState(EMPTY_SCAN_STATE);

  useEffect(() => {
    getLatestScan().then(setScanState);

    function handleStorageChange(changes, areaName) {
      if (areaName !== "local" || !changes[LATEST_SCAN_KEY]) {
        return;
      }

      setScanState(changes[LATEST_SCAN_KEY].newValue || EMPTY_SCAN_STATE);
    }

    chrome.storage.onChanged.addListener(handleStorageChange);

    return () => {
      chrome.storage.onChanged.removeListener(handleStorageChange);
    };
  }, []);

  async function handleUploadCheck() {
    if (!selectedFile) {
      await setLatestScan({
        status: "error",
        error: "Choose an image file first.",
        source: "uploaded file",
      });
      return;
    }

    await setLatestScan({
      status: "loading",
      source: "uploaded file",
    });

    try {
      const result = await detectFile(selectedFile);

      await setLatestScan({
        status: "success",
        result,
        source: "uploaded file",
      });
    } catch (error) {
      await setLatestScan({
        status: "error",
        error: error.message || "Could not check this image.",
        source: "uploaded file",
      });
    }
  }

  const isLoading = scanState.status === "loading";
  const result = scanState.result;

  return (
    <main className="popup">
      <h1>AI Image Detector</h1>

      <label className="field">
        <span>Image file</span>
        <input
          type="file"
          accept="image/jpeg,image/png,image/webp,image/gif"
          onChange={(event) => setSelectedFile(event.target.files?.[0] || null)}
        />
      </label>

      <button type="button" disabled={isLoading} onClick={handleUploadCheck}>
        {isLoading ? "Checking..." : "Check uploaded image"}
      </button>

      <p className={`status status-${scanState.status}`}>
        Status: {scanState.status}
      </p>

      {scanState.error ? <p className="error">{scanState.error}</p> : null}

      {result ? (
        <section className="result" aria-label="Detection result">
          <dl>
            <div>
              <dt>Label</dt>
              <dd>{result.label}</dd>
            </div>
            <div>
              <dt>Confidence</dt>
              <dd>{result.label !== "unknown" ? formatConfidence(result.confidence) : "N/A"}</dd>
            </div>
            <div>
              <dt>Source</dt>
              <dd>{scanState.source || result.input_type}</dd>
            </div>
          </dl>
        </section>
      ) : null}
    </main>
  );
}
