# Verifai - AI Image detector

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

   For Windows users:
      package.json build script used the Unix cp command, it will fail in PowerShell/CMD
      in /extension/package.json, replace
         ```text
           "scripts": {
             "dev": "vite",
             "build": "vite build && cp manifest.json dist/manifest.json",
             "preview": "vite preview"
           },
         ```
      with
         ```text
           "scripts": {
             "dev": "vite",
             "build": "vite build",
             "watch": "vite build --watch",
             "preview": "vite preview"
           },
         ```
      Then move ```manifest.json``` from /extension to /extension/public

4. Open Chrome and go to:

   ```text
   chrome://extensions
   ```

5. Turn on **Developer mode**.

6. Click **Load unpacked** and select the generated folder:

   ```text
   extension/dist
   ```

7. Pin or open **Verifai** from the extensions menu.

## Use the Extension

- To check a local image, open the extension popup, choose an image file, and click **Check uploaded image**.
- To check an image on a webpage, right-click the image and choose **Check if image is AI-generated**.

After making code changes, run `npm run build` again and click the reload button for the extension on `chrome://extensions`.
