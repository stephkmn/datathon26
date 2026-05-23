import { useEffect, useState } from "react";

import { detectFile } from "../api/client.js";
import {
  EMPTY_SCAN_STATE,
  LATEST_SCAN_KEY,
  getLatestScan,
  setLatestScan,
} from "../utils/storage.js";

function formatConfidence(value) {
  if (typeof value !== "number") return "";
  return `${Math.round(value * 100)}%`;
}

function getConfidenceSentence(label, confidence) {
  if (typeof confidence !== "number") return null;
  const pct = Math.round(confidence * 100);
  if (label === "ai-generated") {
    return `We are ${pct}% confident that the given image is AI-generated.`;
  }
  if (label === "real") {
    return `We are ${pct}% confident that the given image is real.`;
  }
  return "";
}

export default function Popup() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [scanState, setScanState] = useState(EMPTY_SCAN_STATE);

  useEffect(() => {
    getLatestScan().then(setScanState);

    function handleStorageChange(changes, areaName) {
      if (areaName !== "local" || !changes[LATEST_SCAN_KEY]) return;
      setScanState(changes[LATEST_SCAN_KEY].newValue || EMPTY_SCAN_STATE);
    }

    chrome.storage.onChanged.addListener(handleStorageChange);
    return () => chrome.storage.onChanged.removeListener(handleStorageChange);
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

    await setLatestScan({ status: "loading", source: "uploaded file" });

    try {
      const result = await detectFile(selectedFile);
      await setLatestScan({ status: "success", result, source: "uploaded file" });
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
  const isUnknown = result?.label === "unknown" || !result?.label;

  return (
    <main className="popup">
      {/* Title */}
      <h1>Verifai</h1>

      {/* Upload row */}
      <div className="upload-row">
        <label className="upload-label">
          {selectedFile ? selectedFile.name : "Upload from computer"}
          <input
            type="file"
            accept="image/jpeg,image/png,image/webp,image/gif"
            onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
          />
        </label>
      </div>

      {/* Check button */}
      <button
        type="button"
        className="check-btn"
        disabled={isLoading}
        onClick={handleUploadCheck}
      >
        {isLoading ? "Checking…" : "Check image"}
      </button>

      {/* Error */}
      {scanState.error && (
        <p className="error-text">{scanState.error}</p>
      )}

      {/* Loading */}
      {isLoading && <p className="loading-text">Analyzing image…</p>}

      {/* Result */}
      {result && !isLoading && (
        <>
          <div className="divider" />
          <div className="result-area" aria-label="Detection result">

            {isUnknown ? (
              <p className="result-label unknown">
                We were unable to classify this image :(
              </p>
            ) : (
              <>
                <p className="result-intro">The given image is most likely:</p>

                <p className="result-label" style={{color: result.label === "ai-generated" ? "#8c4c4c" : "#5d7340"}}>{result.label }</p>

                {result.confidence != null && (
                  <p className="result-confidence">
                    {getConfidenceSentence(result.label, result.confidence)}
                  </p>
                )}
              </>
            )}

            <p className="result-source">
              Image source: <span>{scanState.source || result.input_type || "—"}</span>
            </p>

          </div>
        </>
      )}
    </main>
  );
}