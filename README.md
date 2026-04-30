# Vehicle Monitor WebGPU

This folder contains the browser model rewrite of `vehicle_monitor.py` using ONNX Runtime Web.

## What it does

- runs the exported `yolo26n.onnx` model fully in the browser
- uses WebGPU first, then falls back to ONNX Runtime WASM if WebGPU is unavailable
- supports video file, URL stream, and camera input
- classifies vehicles with the browser model
- prevents duplicate counts with track persistence plus one-time line crossing
- shows annotated output and the 640x640 model input view

## Files

- `index.html`
- `app.css`
- `app.js`
- `models/yolo26n.onnx`

## Run

Serve the folder over HTTP. Camera access and model fetch are more reliable on `localhost` than `file://`.

```powershell
cd "to the vehicle-monitor-ort-web folder"
python -m http.server 8060
```

Then open:

`http://127.0.0.1:8060`

## Notes

- The ONNX model was exported locally from `yolo26n.pt`
- Browser inference uses the exported output shape `(1, 300, 6)` interpreted as `[x1, y1, x2, y2, score, class]`
- For best performance, use a current Chromium-based browser with WebGPU enabled
- Remote URL streams must be browser-playable and must allow CORS if the app needs to read frames from canvas
- HLS `.m3u8` streams are supported through `hls.js`, but the playlist and media segments still need CORS enabled
