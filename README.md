# AI Image Detector

A browser extension that checks whether an image appears to be AI-generated or real. Users can upload an image from the extension popup or right-click an image on a webpage and run detection from the context menu. The model runs locally in the browser using the bundled ONNX files.

## Tech Stack

- React 18 for the extension popup UI
- Vite for local development and production builds
- Chrome Extension Manifest V3
- ONNX Runtime Web for in-browser model inference
- Chrome extension APIs for context menus, storage, background service workers, and the popup
- Optional FastAPI backend prototype in `backend/`

## Run the Extension

1. Install the extension dependencies:

   ```bash
   cd extension
   npm install
   ```

2. Build the extension:

   ```bash
   npm run build
   ```

3. Open Chrome and go to:

   ```text
   chrome://extensions
   ```

4. Turn on **Developer mode**.

5. Click **Load unpacked** and select the generated folder:

   ```text
   extension/dist
   ```

6. Pin or open **AI Image Detector** from the extensions menu.

## Use the Extension

- To check a local image, open the extension popup, choose an image file, and click **Check uploaded image**.
- To check an image on a webpage, right-click the image and choose **Check if image is AI-generated**.

After making code changes, run `npm run build` again and click the reload button for the extension on `chrome://extensions`.
