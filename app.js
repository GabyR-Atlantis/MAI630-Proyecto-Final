const COCO80 = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
  "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
  "scissors", "teddy bear", "hair drier", "toothbrush",
];

const LABEL_ALIASES = new Map([
  ["automobile", "car"],
  ["bike", "motorcycle"],
  ["motorbike", "motorcycle"],
  ["motor cycle", "motorcycle"],
]);

const MODEL_PATH = "./models/yolo26n.onnx";
const MODEL_SIZE = 640;
const HLS_URL_PATTERN = /\.m3u8(?:[?#]|$)/i;

const elements = {
  runtimeStatus: document.getElementById("runtimeStatus"),
  sourceStatus: document.getElementById("sourceStatus"),
  sourceMode: document.getElementById("sourceMode"),
  fileGroup: document.getElementById("fileGroup"),
  urlGroup: document.getElementById("urlGroup"),
  cameraGroup: document.getElementById("cameraGroup"),
  videoFile: document.getElementById("videoFile"),
  streamUrl: document.getElementById("streamUrl"),
  cameraSelect: document.getElementById("cameraSelect"),
  refreshDevices: document.getElementById("refreshDevices"),
  confidenceInput: document.getElementById("confidenceInput"),
  vehicleClassesInput: document.getElementById("vehicleClassesInput"),
  countLineRatioInput: document.getElementById("countLineRatioInput"),
  maxTrackDistanceInput: document.getElementById("maxTrackDistanceInput"),
  maxTrackMissingInput: document.getElementById("maxTrackMissingInput"),
  startButton: document.getElementById("startButton"),
  stopButton: document.getElementById("stopButton"),
  frameStat: document.getElementById("frameStat"),
  liveStat: document.getElementById("liveStat"),
  uniqueStat: document.getElementById("uniqueStat"),
  trackStat: document.getElementById("trackStat"),
  countsList: document.getElementById("countsList"),
  statusDetails: document.getElementById("statusDetails"),
  outputCanvas: document.getElementById("outputCanvas"),
  modelCanvas: document.getElementById("modelCanvas"),
  video: document.getElementById("videoElement"),
};

const outputContext = elements.outputCanvas.getContext("2d");
const modelContext = elements.modelCanvas.getContext("2d");
const frameCanvas = document.createElement("canvas");
frameCanvas.width = MODEL_SIZE;
frameCanvas.height = MODEL_SIZE;
const frameContext = frameCanvas.getContext("2d", { willReadFrequently: true });

let session = null;
let executionProvider = "initializing";
let processing = false;
let activeStream = null;
let activeFileUrl = null;
let activeHls = null;
let animationFrameId = null;
let frameIndex = 0;
let inferencePending = false;
let tracker = null;

class VehicleTracker {
  constructor(maxDistance, maxMissing, countLineRatio) {
    this.maxDistance = maxDistance;
    this.maxMissing = maxMissing;
    this.countLineRatio = countLineRatio;
    this.nextTrackId = 1;
    this.tracks = new Map();
    this.countsByLabel = new Map();
    this.totalCount = 0;
    this.recentCounts = [];
    this.lineY = 0;
  }

  setFrameHeight(height) {
    this.lineY = Math.round(height * this.countLineRatio);
  }

  center(box) {
    return {
      x: box.x + box.width / 2,
      y: box.y + box.height / 2,
    };
  }

  crossedLine(previousCenter, currentCenter) {
    if (!previousCenter) {
      return false;
    }
    return (
      (previousCenter.y < this.lineY && currentCenter.y >= this.lineY) ||
      (previousCenter.y > this.lineY && currentCenter.y <= this.lineY)
    );
  }

  createTrack(detection) {
    const center = this.center(detection.box);
    this.tracks.set(this.nextTrackId, {
      trackId: this.nextTrackId,
      label: detection.label,
      score: detection.score,
      box: detection.box,
      center,
      previousCenter: null,
      missedFrames: 0,
      counted: false,
    });
    this.nextTrackId += 1;
  }

  update(detections) {
    this.recentCounts.push(detections.length);
    this.recentCounts = this.recentCounts.slice(-5);
    const liveCount = this.recentCounts.length ? Math.max(...this.recentCounts) : 0;

    if (!this.tracks.size) {
      detections.forEach((detection) => this.createTrack(detection));
      return this.snapshot(liveCount);
    }

    const unmatchedTrackIds = new Set(this.tracks.keys());
    const unmatchedDetectionIndexes = new Set(detections.map((_, index) => index));
    const pairs = [];

    for (const [trackId, track] of this.tracks.entries()) {
      for (let detectionIndex = 0; detectionIndex < detections.length; detectionIndex += 1) {
        const detection = detections[detectionIndex];
        if (track.label !== detection.label) {
          continue;
        }
        const center = this.center(detection.box);
        const dx = track.center.x - center.x;
        const dy = track.center.y - center.y;
        const distance = Math.hypot(dx, dy);
        if (distance <= this.maxDistance) {
          pairs.push({ distance, trackId, detectionIndex });
        }
      }
    }

    pairs.sort((left, right) => left.distance - right.distance);
    for (const pair of pairs) {
      if (!unmatchedTrackIds.has(pair.trackId) || !unmatchedDetectionIndexes.has(pair.detectionIndex)) {
        continue;
      }

      const detection = detections[pair.detectionIndex];
      const track = this.tracks.get(pair.trackId);
      const center = this.center(detection.box);
      track.previousCenter = track.center;
      track.center = center;
      track.box = detection.box;
      track.label = detection.label;
      track.score = detection.score;
      track.missedFrames = 0;

      if (!track.counted && this.crossedLine(track.previousCenter, track.center)) {
        track.counted = true;
        this.totalCount += 1;
        this.countsByLabel.set(track.label, (this.countsByLabel.get(track.label) || 0) + 1);
      }

      unmatchedTrackIds.delete(pair.trackId);
      unmatchedDetectionIndexes.delete(pair.detectionIndex);
    }

    for (const trackId of unmatchedTrackIds) {
      const track = this.tracks.get(trackId);
      track.missedFrames += 1;
      if (track.missedFrames > this.maxMissing) {
        this.tracks.delete(trackId);
      }
    }

    for (const detectionIndex of unmatchedDetectionIndexes) {
      this.createTrack(detections[detectionIndex]);
    }

    return this.snapshot(liveCount);
  }

  snapshot(liveCount) {
    const activeTracks = [...this.tracks.values()]
      .filter((track) => track.missedFrames === 0)
      .sort((left, right) => left.trackId - right.trackId);

    return {
      lineY: this.lineY,
      activeTracks,
      liveCount,
      uniqueCount: this.totalCount,
      countsByLabel: Object.fromEntries([...this.countsByLabel.entries()].sort()),
    };
  }
}

function normalizeLabel(label) {
  const normalized = label.trim().toLowerCase().replaceAll("_", " ");
  return LABEL_ALIASES.get(normalized) || normalized;
}

function parseVehicleClassFilter() {
  const parsed = elements.vehicleClassesInput.value
    .split(",")
    .map((item) => normalizeLabel(item))
    .filter(Boolean);
  return new Set(parsed.length ? parsed : ["car", "motorcycle", "bus", "truck"]);
}

function setRuntimeStatus(message) {
  elements.runtimeStatus.textContent = message;
}

function setSourceStatus(message) {
  elements.sourceStatus.textContent = message;
}

function updateSourceControls() {
  const mode = elements.sourceMode.value;
  elements.fileGroup.classList.toggle("hidden", mode !== "file");
  elements.urlGroup.classList.toggle("hidden", mode !== "url");
  elements.cameraGroup.classList.toggle("hidden", mode !== "camera");
}

function updateOutputAspectRatio(width, height) {
  if (!width || !height) {
    return;
  }
  elements.outputCanvas.style.setProperty("--video-aspect-ratio", `${width} / ${height}`);
  elements.modelCanvas.style.setProperty("--video-aspect-ratio", `${width} / ${height}`);
}

function computeLetterbox(width, height, targetSize = MODEL_SIZE) {
  const scale = Math.min(targetSize / width, targetSize / height);
  const scaledWidth = Math.round(width * scale);
  const scaledHeight = Math.round(height * scale);
  const offsetX = Math.floor((targetSize - scaledWidth) / 2);
  const offsetY = Math.floor((targetSize - scaledHeight) / 2);

  return {
    scale,
    scaledWidth,
    scaledHeight,
    offsetX,
    offsetY,
  };
}

function destroyHlsSession() {
  if (activeHls) {
    activeHls.destroy();
    activeHls = null;
  }
}

function resetVideoElement() {
  elements.video.pause();
  elements.video.removeAttribute("src");
  elements.video.removeAttribute("crossorigin");
  elements.video.srcObject = null;
  elements.video.load();
}

function getMediaErrorMessage(sourceUrl = "") {
  const mediaError = elements.video.error;
  const sourceHint = sourceUrl ? ` Source: ${sourceUrl}` : "";
  if (!mediaError) {
    return `The selected media source could not be played in this browser.${sourceHint}`;
  }

  switch (mediaError.code) {
    case MediaError.MEDIA_ERR_ABORTED:
      return `Media loading was interrupted before playback started.${sourceHint}`;
    case MediaError.MEDIA_ERR_NETWORK:
      return `The browser could not load the media stream. Check the URL and server availability.${sourceHint}`;
    case MediaError.MEDIA_ERR_DECODE:
      return `The media source loaded but could not be decoded by this browser.${sourceHint}`;
    case MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED:
      return `The media format or stream protocol is not supported here. For remote streams, use a browser-playable MP4/WebM URL or an HLS .m3u8 stream with CORS enabled.${sourceHint}`;
    default:
      return `The selected media source could not be played in this browser.${sourceHint}`;
  }
}

function assertVideoFrameReadable(sourceUrl = "") {
  try {
    frameContext.drawImage(elements.video, 0, 0, 1, 1);
    frameContext.getImageData(0, 0, 1, 1);
  } catch (error) {
    if (error instanceof DOMException && error.name === "SecurityError") {
      throw new Error(
        `This stream is cross-origin and cannot be analyzed because the server does not allow canvas access. Configure CORS for the media origin or serve the stream from the same origin. Source: ${sourceUrl || "remote stream"}`,
      );
    }
    throw error;
  }
}

function renderCounts(countsByLabel) {
  elements.countsList.innerHTML = "";
  const entries = Object.entries(countsByLabel);
  if (!entries.length) {
    const chip = document.createElement("div");
    chip.className = "count-chip";
    chip.textContent = "No vehicles counted yet";
    elements.countsList.appendChild(chip);
    return;
  }

  entries
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .forEach(([label, count]) => {
      const chip = document.createElement("div");
      chip.className = "count-chip";
      chip.textContent = `${label}: ${count}`;
      elements.countsList.appendChild(chip);
    });
}

function renderStatus(state) {
  elements.frameStat.textContent = String(frameIndex);
  elements.liveStat.textContent = String(state.liveCount);
  elements.uniqueStat.textContent = String(state.uniqueCount);
  elements.trackStat.textContent = String(state.activeTracks.length);
  renderCounts(state.countsByLabel);
  elements.statusDetails.textContent = [
    `provider: ${executionProvider}`,
    `model: ${MODEL_PATH}`,
    `line_y: ${state.lineY}`,
    `active_tracks: ${state.activeTracks.length}`,
    `counts: ${JSON.stringify(state.countsByLabel)}`,
  ].join("\n");
}

async function waitForRuntime() {
  if (window.ort?.InferenceSession) {
    return;
  }
  await new Promise((resolve, reject) => {
    const deadline = performance.now() + 15000;
    const poll = () => {
      if (window.ort?.InferenceSession) {
        resolve();
        return;
      }
      if (performance.now() > deadline) {
        reject(new Error("ONNX Runtime Web did not load."));
        return;
      }
      window.setTimeout(poll, 100);
    };
    poll();
  });
}

async function createSession() {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  ort.env.wasm.numThreads = 1;

  try {
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
    });
    executionProvider = "webgpu";
  } catch (webgpuError) {
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    executionProvider = "wasm";
  }
}

async function refreshCameraList() {
  if (!navigator.mediaDevices?.enumerateDevices) {
    setSourceStatus("Camera listing is unavailable in this browser.");
    return;
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter((device) => device.kind === "videoinput");
  elements.cameraSelect.innerHTML = "";

  if (!cameras.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No camera found";
    elements.cameraSelect.appendChild(option);
    return;
  }

  cameras.forEach((device, index) => {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || `Camera ${index + 1}`;
    elements.cameraSelect.appendChild(option);
  });
}

function stopMediaResources() {
  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  destroyHlsSession();
  if (activeStream) {
    activeStream.getTracks().forEach((track) => track.stop());
    activeStream = null;
  }
  if (activeFileUrl) {
    URL.revokeObjectURL(activeFileUrl);
    activeFileUrl = null;
  }
  resetVideoElement();
}

function stopProcessing(reason = "Stopped") {
  processing = false;
  inferencePending = false;
  stopMediaResources();
  elements.startButton.disabled = false;
  elements.stopButton.disabled = true;
  setSourceStatus(reason);
}

async function waitForPlayableVideo(callLoad = true, sourceUrl = "") {
  await new Promise((resolve, reject) => {
    const onLoadedData = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error(getMediaErrorMessage(sourceUrl)));
    };
    const cleanup = () => {
      elements.video.removeEventListener("loadeddata", onLoadedData);
      elements.video.removeEventListener("error", onError);
    };
    if (elements.video.readyState >= 2 && elements.video.videoWidth && elements.video.videoHeight) {
      resolve();
      return;
    }
    elements.video.addEventListener("loadeddata", onLoadedData);
    elements.video.addEventListener("error", onError);
    if (callLoad) {
      elements.video.load();
    }
  });
  try {
    await elements.video.play();
  } catch (error) {
    throw new Error(getMediaErrorMessage(sourceUrl));
  }
}

async function startFromFile() {
  const [file] = elements.videoFile.files;
  if (!file) {
    throw new Error("Select a video file first.");
  }
  resetVideoElement();
  activeFileUrl = URL.createObjectURL(file);
  elements.video.src = activeFileUrl;
  await waitForPlayableVideo(true, file.name);
}

async function loadRemoteVideoAsBlob(sourceUrl) {
  const response = await fetch(sourceUrl, { mode: "cors" });
  if (!response.ok) {
    throw new Error(`The stream server returned HTTP ${response.status}.`);
  }
  const blob = await response.blob();
  activeFileUrl = URL.createObjectURL(blob);
  elements.video.src = activeFileUrl;
  await waitForPlayableVideo(true, sourceUrl);
}

async function loadDirectVideoUrl(sourceUrl) {
  elements.video.crossOrigin = "anonymous";
  elements.video.src = sourceUrl;
  await waitForPlayableVideo(true, sourceUrl);
  assertVideoFrameReadable(sourceUrl);
}

async function loadHlsStream(sourceUrl) {
  resetVideoElement();
  elements.video.crossOrigin = "anonymous";

  if (elements.video.canPlayType("application/vnd.apple.mpegurl")) {
    elements.video.src = sourceUrl;
    await waitForPlayableVideo(true, sourceUrl);
    assertVideoFrameReadable(sourceUrl);
    return;
  }

  if (!window.Hls?.isSupported?.()) {
    throw new Error(`This browser cannot play HLS streams here. Use a direct MP4/WebM URL or a browser with HLS support. Source: ${sourceUrl}`);
  }

  activeHls = new window.Hls({
    enableWorker: true,
    lowLatencyMode: true,
  });

  await new Promise((resolve, reject) => {
    const cleanup = () => {
      activeHls.off(window.Hls.Events.MANIFEST_PARSED, onParsed);
      activeHls.off(window.Hls.Events.ERROR, onError);
    };
    const onParsed = () => {
      cleanup();
      resolve();
    };
    const onError = (_event, data) => {
      if (!data?.fatal) {
        return;
      }
      cleanup();
      reject(new Error(`The HLS stream could not be loaded. Ensure the playlist and segments are reachable with CORS enabled. Source: ${sourceUrl}`));
    };

    activeHls.on(window.Hls.Events.MANIFEST_PARSED, onParsed);
    activeHls.on(window.Hls.Events.ERROR, onError);
    activeHls.attachMedia(elements.video);
    activeHls.loadSource(sourceUrl);
  });

  await waitForPlayableVideo(false, sourceUrl);
  assertVideoFrameReadable(sourceUrl);
}

async function startFromUrl() {
  const rawUrl = elements.streamUrl.value.trim();
  if (!rawUrl) {
    throw new Error("Enter a stream URL first.");
  }
  const sourceUrl = new URL(rawUrl, window.location.href).toString();

  if (HLS_URL_PATTERN.test(sourceUrl)) {
    await loadHlsStream(sourceUrl);
    return;
  }

  resetVideoElement();
  try {
    await loadRemoteVideoAsBlob(sourceUrl);
  } catch (_fetchError) {
    await loadDirectVideoUrl(sourceUrl);
  }
}

async function startFromCamera() {
  resetVideoElement();
  const deviceId = elements.cameraSelect.value;
  const constraints = deviceId
    ? { video: { deviceId: { exact: deviceId } }, audio: false }
    : { video: true, audio: false };
  activeStream = await navigator.mediaDevices.getUserMedia(constraints);
  elements.video.srcObject = activeStream;
  await waitForPlayableVideo(false, "camera");
}

function preprocessFrame() {
  try {
    const sourceWidth = elements.video.videoWidth || MODEL_SIZE;
    const sourceHeight = elements.video.videoHeight || MODEL_SIZE;
    const letterbox = computeLetterbox(sourceWidth, sourceHeight);

    if (elements.modelCanvas.width !== sourceWidth || elements.modelCanvas.height !== sourceHeight) {
      elements.modelCanvas.width = sourceWidth;
      elements.modelCanvas.height = sourceHeight;
    }
    modelContext.drawImage(elements.video, 0, 0, sourceWidth, sourceHeight);

    frameContext.fillStyle = "#000000";
    frameContext.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
    frameContext.drawImage(
      elements.video,
      0,
      0,
      sourceWidth,
      sourceHeight,
      letterbox.offsetX,
      letterbox.offsetY,
      letterbox.scaledWidth,
      letterbox.scaledHeight,
    );
    const imageData = frameContext.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
    const { data } = imageData;
    const pixels = new Float32Array(1 * 3 * MODEL_SIZE * MODEL_SIZE);
    const planeSize = MODEL_SIZE * MODEL_SIZE;

    for (let index = 0; index < planeSize; index += 1) {
      const dataIndex = index * 4;
      pixels[index] = data[dataIndex] / 255;
      pixels[planeSize + index] = data[dataIndex + 1] / 255;
      pixels[planeSize * 2 + index] = data[dataIndex + 2] / 255;
    }

    return new ort.Tensor("float32", pixels, [1, 3, MODEL_SIZE, MODEL_SIZE]);
  } catch (error) {
    if (error instanceof DOMException && error.name === "SecurityError") {
      throw new Error("The current video source is blocking canvas reads. For URL streams, the media server must allow CORS.");
    }
    throw error;
  }
}

function postprocessOutput(outputTensor, originalWidth, originalHeight, confidenceThreshold, allowedLabels) {
  const detections = [];
  const data = outputTensor.data;
  const letterbox = computeLetterbox(originalWidth, originalHeight);

  for (let offset = 0; offset < data.length; offset += 6) {
    const score = data[offset + 4];
    if (!Number.isFinite(score) || score < confidenceThreshold) {
      continue;
    }

    const classId = Math.round(data[offset + 5]);
    const rawLabel = COCO80[classId];
    if (!rawLabel) {
      continue;
    }

    const label = normalizeLabel(rawLabel);
    if (!allowedLabels.has(label)) {
      continue;
    }

    const x1 = Math.max(
      0,
      Math.min(originalWidth, (data[offset] - letterbox.offsetX) / letterbox.scale),
    );
    const y1 = Math.max(
      0,
      Math.min(originalHeight, (data[offset + 1] - letterbox.offsetY) / letterbox.scale),
    );
    const x2 = Math.max(
      0,
      Math.min(originalWidth, (data[offset + 2] - letterbox.offsetX) / letterbox.scale),
    );
    const y2 = Math.max(
      0,
      Math.min(originalHeight, (data[offset + 3] - letterbox.offsetY) / letterbox.scale),
    );
    const width = Math.max(0, x2 - x1);
    const height = Math.max(0, y2 - y1);
    if (width < 4 || height < 4) {
      continue;
    }

    detections.push({
      label,
      score,
      box: { x: x1, y: y1, width, height },
    });
  }

  return detections;
}

function drawAnnotatedFrame(state) {
  const width = elements.video.videoWidth || 960;
  const height = elements.video.videoHeight || 540;
  if (elements.outputCanvas.width !== width || elements.outputCanvas.height !== height) {
    elements.outputCanvas.width = width;
    elements.outputCanvas.height = height;
  }
  updateOutputAspectRatio(width, height);

  outputContext.drawImage(elements.video, 0, 0, width, height);
  outputContext.strokeStyle = "#ffb400";
  outputContext.lineWidth = 2;
  outputContext.beginPath();
  outputContext.moveTo(0, state.lineY);
  outputContext.lineTo(width, state.lineY);
  outputContext.stroke();

  outputContext.strokeStyle = "#00c86d";
  outputContext.fillStyle = "#00c86d";
  outputContext.font = "16px Inter, Segoe UI, Arial";

  for (const track of state.activeTracks) {
    const { x, y, width: boxWidth, height: boxHeight } = track.box;
    outputContext.strokeRect(x, y, boxWidth, boxHeight);
    let label = `id=${track.trackId} ${track.label}`;
    if (track.score !== null && track.score !== undefined) {
      label += ` ${track.score.toFixed(2)}`;
    }
    if (track.counted) {
      label += " counted";
    }
    outputContext.fillText(label, x, Math.max(18, y - 8));
  }

  outputContext.fillStyle = "#ffe36e";
  outputContext.font = "20px Inter, Segoe UI, Arial";
  outputContext.fillText(
    `frame=${frameIndex} live=${state.liveCount} unique=${state.uniqueCount}`,
    15,
    30,
  );

  const summaryEntries = Object.entries(state.countsByLabel);
  const summary = summaryEntries.length
    ? "counted=" + summaryEntries.map(([label, count]) => `${label}:${count}`).join(", ")
    : "counted=0";
  outputContext.fillText(summary, 15, 58);
}

async function runInferenceStep() {
  if (!processing || inferencePending || elements.video.readyState < 2) {
    return;
  }
  inferencePending = true;

  try {
    const input = preprocessFrame();
    const outputMap = await session.run({ images: input });
    const output = outputMap[session.outputNames[0]];
    const detections = postprocessOutput(
      output,
      elements.video.videoWidth,
      elements.video.videoHeight,
      Number.parseFloat(elements.confidenceInput.value) || 0.35,
      parseVehicleClassFilter(),
    );
    const state = tracker.update(detections);
    drawAnnotatedFrame(state);
    renderStatus(state);
    frameIndex += 1;
  } finally {
    inferencePending = false;
  }
}

function processLoop() {
  if (!processing) {
    return;
  }
  runInferenceStep().catch((error) => {
    console.error(error);
    stopProcessing(error.message);
  });
  animationFrameId = requestAnimationFrame(processLoop);
}

async function startProcessing() {
  if (!session) {
    throw new Error("The runtime is still loading.");
  }

  stopProcessing("Reinitializing");
  updateSourceControls();
  frameIndex = 0;

  if (elements.sourceMode.value === "camera") {
    await startFromCamera();
  } else if (elements.sourceMode.value === "url") {
    await startFromUrl();
  } else {
    await startFromFile();
  }

  tracker = new VehicleTracker(
    Number.parseFloat(elements.maxTrackDistanceInput.value) || 90,
    Number.parseInt(elements.maxTrackMissingInput.value, 10) || 12,
    Number.parseFloat(elements.countLineRatioInput.value) || 0.5,
  );
  tracker.setFrameHeight(elements.video.videoHeight || 540);
  renderCounts({});

  processing = true;
  elements.startButton.disabled = true;
  elements.stopButton.disabled = false;
  if (elements.sourceMode.value === "camera") {
    setSourceStatus("Camera active");
  } else if (elements.sourceMode.value === "url") {
    setSourceStatus("URL stream active");
  } else {
    setSourceStatus("Video active");
  }
  updateOutputAspectRatio(elements.video.videoWidth || 960, elements.video.videoHeight || 540);
  processLoop();
}

function bindEvents() {
  elements.sourceMode.addEventListener("change", updateSourceControls);
  elements.refreshDevices.addEventListener("click", async () => {
    try {
      await refreshCameraList();
    } catch (error) {
      setSourceStatus(error.message);
    }
  });
  elements.startButton.addEventListener("click", async () => {
    try {
      await startProcessing();
    } catch (error) {
      setSourceStatus(error.message);
    }
  });
  elements.stopButton.addEventListener("click", () => {
    stopProcessing("Stopped by user.");
  });
  elements.video.addEventListener("ended", () => {
    if (processing && elements.sourceMode.value === "file") {
      stopProcessing("Video completed.");
    }
  });
}

async function bootstrap() {
  updateSourceControls();
  bindEvents();
  await waitForRuntime();
  await createSession();
  await refreshCameraList();
  renderCounts({});
  setRuntimeStatus(`Runtime ready (${executionProvider})`);
  setSourceStatus("Ready");
}

bootstrap().catch((error) => {
  console.error(error);
  setRuntimeStatus(error.message);
  setSourceStatus("Blocked");
});
