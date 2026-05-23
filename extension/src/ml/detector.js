import * as ort from "onnxruntime-web/wasm";

const MODEL_VERSION = "v1.0";
const MODEL_SIZE = 32;
const AI_THRESHOLD = 0.25;
const REAL_THRESHOLD = 0.75;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let sessionPromise = null;

function getRuntimeUrl(path) {
  return chrome.runtime.getURL(path);
}

function getSession() {
  if (!sessionPromise) {
    ort.env.wasm.numThreads = 1;

    sessionPromise = ort.InferenceSession.create(getRuntimeUrl("model2.onnx"), {
      executionProviders: ["wasm"],
      externalData: [
        {
          path: "model2.onnx.data",
          data: getRuntimeUrl("model2.onnx.data"),
        },
      ],
    });
  }

  return sessionPromise;
}

function createCanvas(width, height) {
  if (typeof OffscreenCanvas !== "undefined") {
    return new OffscreenCanvas(width, height);
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

async function blobToImageBitmap(blob) {
  try {
    return await createImageBitmap(blob);
  } catch {
    throw new Error("Could not decode this image.");
  }
}

function imageBitmapToTensor(imageBitmap) {
  const canvas = createCanvas(MODEL_SIZE, MODEL_SIZE);
  const context = canvas.getContext("2d", { willReadFrequently: true });

  if (!context) {
    throw new Error("Could not prepare image for analysis.");
  }

  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "medium";
  context.clearRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  context.drawImage(imageBitmap, 0, 0, MODEL_SIZE, MODEL_SIZE);

  const { data } = context.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
  const planeSize = MODEL_SIZE * MODEL_SIZE;
  const input = new Float32Array(3 * planeSize);

  for (let pixel = 0; pixel < planeSize; pixel += 1) {
    const source = pixel * 4;
    const red = data[source] / 255;
    const green = data[source + 1] / 255;
    const blue = data[source + 2] / 255;

    input[pixel] = (red - MEAN[0]) / STD[0];
    input[planeSize + pixel] = (green - MEAN[1]) / STD[1];
    input[2 * planeSize + pixel] = (blue - MEAN[2]) / STD[2];
  }

  return new ort.Tensor("float32", input, [1, 3, MODEL_SIZE, MODEL_SIZE]);
}

function deriveLabelAndConfidence(probability) {
  if (probability <= AI_THRESHOLD) {
    return {
      label: "ai-generated",
      confidence: 1.0 - probability / AI_THRESHOLD,
    };
  }

  if (probability >= REAL_THRESHOLD) {
    return {
      label: "real",
      confidence: (probability - REAL_THRESHOLD) / (1.0 - REAL_THRESHOLD),
    };
  }

  return {
    label: "unknown",
    confidence: (probability - AI_THRESHOLD) / AI_THRESHOLD,
  };
}

function buildResult(probability, inputType) {
  const clampedProbability = Math.max(0, Math.min(1, Number(probability)));
  const { label, confidence } = deriveLabelAndConfidence(clampedProbability);

  return {
    label,
    confidence: Math.round(Math.max(0, Math.min(1, confidence)) * 100) / 100,
    explanation: "Analyzed locally in the browser.",
    model_version: MODEL_VERSION,
    input_type: inputType,
  };
}

export async function detectBlob(blob, inputType) {
  const imageBitmap = await blobToImageBitmap(blob);

  try {
    const session = await getSession();
    const tensor = imageBitmapToTensor(imageBitmap);
    const feeds = {
      [session.inputNames[0]]: tensor,
    };
    const outputs = await session.run(feeds);
    const output = outputs[session.outputNames[0]];

    return buildResult(output.data[0], inputType);
  } finally {
    imageBitmap.close?.();
  }
}

export function detectFile(file) {
  return detectBlob(file, "file");
}

export async function detectUrl(imageUrl) {
  const response = await fetch(imageUrl);

  if (!response.ok) {
    throw new Error("Could not fetch image from the selected URL.");
  }

  const blob = await response.blob();
  const contentType = response.headers.get("content-type") || blob.type;

  if (contentType && !contentType.toLowerCase().startsWith("image/")) {
    throw new Error("The selected URL did not return an image.");
  }

  return detectBlob(blob, "url");
}
