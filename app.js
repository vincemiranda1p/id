const CARD_ASPECT_RATIO = 1.586;
const ANALYSIS_SCALE = 0.42;
const ID_INTERVAL_MS = 120;
const FACE_INTERVAL_MS = 480;
const UI_INTERVAL_MS = 90;

const BLUR_LAPLACIAN_THRESHOLD = 30;
const FOCUS_TENENGRAD_THRESHOLD = 22;
const FACE_BLUR_LAPLACIAN_THRESHOLD_DEFAULT = 6;

const faceBlurThresholdParam = Number(new URLSearchParams(window.location.search).get("faceBlurThreshold"));
const FACE_BLUR_LAPLACIAN_THRESHOLD =
  Number.isFinite(faceBlurThresholdParam) && faceBlurThresholdParam > 0
    ? faceBlurThresholdParam
    : FACE_BLUR_LAPLACIAN_THRESHOLD_DEFAULT;

const ISSUE_NO_ID = "NO ID FOUND";
const ISSUE_ID_NOT_IN_FRAME = "ID NOT IN FRAME";
const ISSUE_WHOLE_ID_BLUR = "BLURRED / UNCLEAR PHOTO OF WHOLE ID";
const ISSUE_BLURRY_FACE = "BLURRY FACE";
const ISSUE_NO_FACE = "NO FACE";

const STATE_TRANSITION_DELAY_MS = {
  whiteToRed: 180,
  whiteToGreen: 260,
  redToGreen: 320,
  redToWhite: 360,
  greenToRed: 780,
  greenToWhite: 620,
};

const video = document.getElementById("camera");
const guide = document.getElementById("guide");
const faceBoxElement = document.getElementById("faceBox");
const faceConfidenceElement = document.getElementById("faceConfidence");

const statusLabel = document.getElementById("statusLabel");
const issuesList = document.getElementById("issuesList");
const submitButton = document.getElementById("submitButton");
const redoButton = document.getElementById("redoButton");

const captureResult = document.getElementById("captureResult");
const capturePreview = document.getElementById("capturePreview");
const downloadLink = document.getElementById("downloadLink");

const analysisCanvas = document.getElementById("analysisCanvas");
const analysisCtx = analysisCanvas.getContext("2d", { willReadFrequently: true });

const faceCanvas = document.getElementById("faceCanvas");
const faceCtx = faceCanvas.getContext("2d", { willReadFrequently: true });

const captureCanvas = document.getElementById("captureCanvas");
const captureCtx = captureCanvas.getContext("2d");
const faceSharpnessCanvas = document.createElement("canvas");
const faceSharpnessCtx = faceSharpnessCanvas.getContext("2d", { willReadFrequently: true });

let faceDetector = null;
let activeStream = null;
let cvReady = false;
let cameraReady = false;
let loopHandle = null;
let scanLocked = false;

const state = {
  idDetected: false,
  fitsGuide: false,
  blurred: false,
  outOfFocus: false,
  wholeIdUnclear: false,
  faceDetected: false,
  faceBlurry: false,
  faceConfidence: 0,
  faceBoxInAnalysis: null,
  lastGuideRect: null,
  issues: [ISSUE_NO_ID],
  laplacianVariance: 0,
  tenengrad: 0,
  faceLaplacianVariance: 0,
};

const timing = {
  lastIdAt: 0,
  lastFaceAt: 0,
  lastUiAt: 0,
};

const uiStability = {
  color: "white",
  issues: [ISSUE_NO_ID],
  candidateColor: "white",
  candidateIssues: [ISSUE_NO_ID],
  candidateSince: 0,
};

function clampRect(rect, width, height) {
  const x = Math.max(0, Math.min(rect.x, width - 1));
  const y = Math.max(0, Math.min(rect.y, height - 1));
  const w = Math.max(1, Math.min(rect.width, width - x));
  const h = Math.max(1, Math.min(rect.height, height - y));
  return { x, y, width: w, height: h };
}

function setButtonState(button, enabled) {
  button.disabled = !enabled;
  button.classList.toggle("enabled", enabled);
}

function getVideoCoverCrop(videoWidth, videoHeight, renderWidth, renderHeight) {
  const videoAspect = videoWidth / videoHeight;
  const renderAspect = renderWidth / renderHeight;

  if (videoAspect > renderAspect) {
    const sourceHeight = videoHeight;
    const sourceWidth = sourceHeight * renderAspect;
    const sourceX = (videoWidth - sourceWidth) / 2;
    return { sx: sourceX, sy: 0, sw: sourceWidth, sh: sourceHeight };
  }

  const sourceWidth = videoWidth;
  const sourceHeight = sourceWidth / renderAspect;
  const sourceY = (videoHeight - sourceHeight) / 2;
  return { sx: 0, sy: sourceY, sw: sourceWidth, sh: sourceHeight };
}

function ensureAnalysisSize() {
  const vw = Math.max(1, Math.round(video.clientWidth * ANALYSIS_SCALE));
  const vh = Math.max(1, Math.round(video.clientHeight * ANALYSIS_SCALE));

  if (analysisCanvas.width !== vw || analysisCanvas.height !== vh) {
    analysisCanvas.width = vw;
    analysisCanvas.height = vh;
  }
}

function getGuideRectInAnalysisSpace() {
  const videoRect = video.getBoundingClientRect();
  const guideRect = guide.getBoundingClientRect();

  const x = (guideRect.left - videoRect.left) * ANALYSIS_SCALE;
  const y = (guideRect.top - videoRect.top) * ANALYSIS_SCALE;
  const width = guideRect.width * ANALYSIS_SCALE;
  const height = guideRect.height * ANALYSIS_SCALE;

  return clampRect(
    {
      x: Math.round(x),
      y: Math.round(y),
      width: Math.round(width),
      height: Math.round(height),
    },
    analysisCanvas.width,
    analysisCanvas.height
  );
}

function drawCurrentFrameToAnalysisCanvas() {
  ensureAnalysisSize();

  const renderWidth = video.clientWidth;
  const renderHeight = video.clientHeight;
  const crop = getVideoCoverCrop(video.videoWidth, video.videoHeight, renderWidth, renderHeight);

  analysisCtx.drawImage(
    video,
    crop.sx,
    crop.sy,
    crop.sw,
    crop.sh,
    0,
    0,
    analysisCanvas.width,
    analysisCanvas.height
  );
}

function iou(rectA, rectB) {
  const x1 = Math.max(rectA.x, rectB.x);
  const y1 = Math.max(rectA.y, rectB.y);
  const x2 = Math.min(rectA.x + rectA.width, rectB.x + rectB.width);
  const y2 = Math.min(rectA.y + rectA.height, rectB.y + rectB.height);

  if (x2 <= x1 || y2 <= y1) {
    return 0;
  }

  const intersection = (x2 - x1) * (y2 - y1);
  const union = rectA.width * rectA.height + rectB.width * rectB.height - intersection;
  return union > 0 ? intersection / union : 0;
}

function detectIdRectangle(gray, guideRect) {
  const blur = new cv.Mat();
  const edges = new cv.Mat();
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();

  try {
    cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0);
    cv.Canny(blur, edges, 70, 210);
    cv.morphologyEx(edges, edges, cv.MORPH_CLOSE, kernel);

    cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    let best = null;
    const minContourArea = guideRect.width * guideRect.height * 0.35;

    for (let i = 0; i < contours.size(); i += 1) {
      const contour = contours.get(i);
      const perimeter = cv.arcLength(contour, true);
      const approx = new cv.Mat();

      cv.approxPolyDP(contour, approx, 0.02 * perimeter, true);
      const area = cv.contourArea(approx);

      if (approx.rows === 4 && area > minContourArea) {
        const rect = cv.boundingRect(approx);
        const ratio = rect.width / rect.height;

        const ratioError = Math.abs(ratio - CARD_ASPECT_RATIO) / CARD_ASPECT_RATIO;
        if (ratioError <= 0.22) {
          const overlap = iou(rect, guideRect);
          const overlapW = Math.max(
            0,
            Math.min(rect.x + rect.width, guideRect.x + guideRect.width) - Math.max(rect.x, guideRect.x)
          );
          const overlapH = Math.max(
            0,
            Math.min(rect.y + rect.height, guideRect.y + guideRect.height) - Math.max(rect.y, guideRect.y)
          );
          const normalizedCoverage = (overlapW * overlapH) / (guideRect.width * guideRect.height);

          const score = overlap * 0.7 + normalizedCoverage * 0.3;
          if (!best || score > best.score) {
            best = { rect, score, overlap, ratio };
          }
        }
      }

      contour.delete();
      approx.delete();
    }

    if (!best) {
      return { detected: false, fits: false };
    }

    const widthError = Math.abs(best.rect.width - guideRect.width) / guideRect.width;
    const heightError = Math.abs(best.rect.height - guideRect.height) / guideRect.height;
    const centerDeltaX =
      Math.abs(best.rect.x + best.rect.width / 2 - (guideRect.x + guideRect.width / 2)) / guideRect.width;
    const centerDeltaY =
      Math.abs(best.rect.y + best.rect.height / 2 - (guideRect.y + guideRect.height / 2)) / guideRect.height;

    const fits =
      best.overlap >= 0.72 &&
      widthError <= 0.14 &&
      heightError <= 0.14 &&
      centerDeltaX <= 0.09 &&
      centerDeltaY <= 0.09;

    return {
      detected: true,
      fits,
      rect: best.rect,
      overlap: best.overlap,
    };
  } finally {
    blur.delete();
    edges.delete();
    kernel.delete();
    contours.delete();
    hierarchy.delete();
  }
}

function computeSharpnessFromGrayRoi(gray, roiRect) {
  const safeRect = clampRect(roiRect, gray.cols, gray.rows);
  const rect = new cv.Rect(safeRect.x, safeRect.y, safeRect.width, safeRect.height);
  const roi = gray.roi(rect);

  const lap = new cv.Mat();
  const mean = new cv.Mat();
  const stddev = new cv.Mat();

  const gradX = new cv.Mat();
  const gradY = new cv.Mat();
  const magnitude = new cv.Mat();

  try {
    cv.Laplacian(roi, lap, cv.CV_64F);
    cv.meanStdDev(lap, mean, stddev);
    const lapVariance = stddev.doubleAt(0, 0) ** 2;

    cv.Sobel(roi, gradX, cv.CV_32F, 1, 0, 3);
    cv.Sobel(roi, gradY, cv.CV_32F, 0, 1, 3);
    cv.magnitude(gradX, gradY, magnitude);
    const tenengrad = cv.mean(magnitude)[0];

    return {
      lapVariance,
      tenengrad,
      blurred: lapVariance < BLUR_LAPLACIAN_THRESHOLD,
      outOfFocus: tenengrad < FOCUS_TENENGRAD_THRESHOLD,
    };
  } finally {
    roi.delete();
    lap.delete();
    mean.delete();
    stddev.delete();
    gradX.delete();
    gradY.delete();
    magnitude.delete();
  }
}

function computeFaceLaplacianVariance(faceBox) {
  if (!faceBox) {
    return 0;
  }

  const faceRect = clampRect(
    {
      x: Math.round(faceBox.originX),
      y: Math.round(faceBox.originY),
      width: Math.round(faceBox.width),
      height: Math.round(faceBox.height),
    },
    faceCanvas.width,
    faceCanvas.height
  );

  if (faceRect.width < 24 || faceRect.height < 24) {
    return 0;
  }

  const targetW = Math.max(24, Math.min(88, faceRect.width));
  const targetH = Math.max(24, Math.min(88, faceRect.height));

  faceSharpnessCanvas.width = targetW;
  faceSharpnessCanvas.height = targetH;
  faceSharpnessCtx.drawImage(
    faceCanvas,
    faceRect.x,
    faceRect.y,
    faceRect.width,
    faceRect.height,
    0,
    0,
    targetW,
    targetH
  );

  const pixels = faceSharpnessCtx.getImageData(0, 0, targetW, targetH).data;
  const luma = new Float32Array(targetW * targetH);

  for (let i = 0, p = 0; i < luma.length; i += 1, p += 4) {
    luma[i] = pixels[p] * 0.299 + pixels[p + 1] * 0.587 + pixels[p + 2] * 0.114;
  }

  let sum = 0;
  let sumSq = 0;
  let count = 0;

  for (let y = 1; y < targetH - 1; y += 1) {
    const row = y * targetW;
    for (let x = 1; x < targetW - 1; x += 1) {
      const idx = row + x;
      const lap =
        4 * luma[idx] - luma[idx - 1] - luma[idx + 1] - luma[idx - targetW] - luma[idx + targetW];
      sum += lap;
      sumSq += lap * lap;
      count += 1;
    }
  }

  if (count === 0) {
    return 0;
  }

  const mean = sum / count;
  return sumSq / count - mean * mean;
}

function getFaceConfidence(bestFace) {
  if (!bestFace) {
    return 0;
  }

  const categoryScore = bestFace.categories?.[0]?.score;
  if (typeof categoryScore === "number") {
    return categoryScore;
  }

  const score = bestFace.score;
  if (typeof score === "number") {
    return score;
  }

  return 0;
}

function clearFaceOverlay() {
  faceBoxElement.classList.add("hidden");
  faceBoxElement.style.left = "0px";
  faceBoxElement.style.top = "0px";
  faceBoxElement.style.width = "0px";
  faceBoxElement.style.height = "0px";
  faceConfidenceElement.textContent = "Face 0%";
}

function renderFaceOverlay() {
  if (!cameraReady || scanLocked || !state.faceDetected || !state.faceBoxInAnalysis) {
    clearFaceOverlay();
    return;
  }

  if (!analysisCanvas.width || !analysisCanvas.height || !video.clientWidth || !video.clientHeight) {
    clearFaceOverlay();
    return;
  }

  const scaleX = video.clientWidth / analysisCanvas.width;
  const scaleY = video.clientHeight / analysisCanvas.height;

  const viewX = state.faceBoxInAnalysis.x * scaleX;
  const viewY = state.faceBoxInAnalysis.y * scaleY;
  const viewW = state.faceBoxInAnalysis.width * scaleX;
  const viewH = state.faceBoxInAnalysis.height * scaleY;

  faceBoxElement.style.left = `${viewX}px`;
  faceBoxElement.style.top = `${viewY}px`;
  faceBoxElement.style.width = `${viewW}px`;
  faceBoxElement.style.height = `${viewH}px`;
  faceConfidenceElement.textContent = `Face ${(state.faceConfidence * 100).toFixed(1)}%`;
  faceBoxElement.classList.remove("hidden");
}

function updateFaceDetection(guideRect) {
  if (!faceDetector) {
    state.faceDetected = false;
    state.faceBlurry = false;
    state.faceLaplacianVariance = 0;
    state.faceConfidence = 0;
    state.faceBoxInAnalysis = null;
    return;
  }

  faceCanvas.width = guideRect.width;
  faceCanvas.height = guideRect.height;
  faceCtx.drawImage(
    analysisCanvas,
    guideRect.x,
    guideRect.y,
    guideRect.width,
    guideRect.height,
    0,
    0,
    faceCanvas.width,
    faceCanvas.height
  );

  const detection = faceDetector.detect(faceCanvas);
  const detections = detection?.detections ?? [];

  if (detections.length === 0) {
    state.faceDetected = false;
    state.faceBlurry = false;
    state.faceLaplacianVariance = 0;
    state.faceConfidence = 0;
    state.faceBoxInAnalysis = null;
    return;
  }

  let bestFace = detections[0];
  let bestArea = bestFace.boundingBox.width * bestFace.boundingBox.height;

  for (let i = 1; i < detections.length; i += 1) {
    const candidate = detections[i];
    const area = candidate.boundingBox.width * candidate.boundingBox.height;
    if (area > bestArea) {
      bestArea = area;
      bestFace = candidate;
    }
  }

  const bestBox = clampRect(
    {
      x: Math.round(bestFace.boundingBox.originX),
      y: Math.round(bestFace.boundingBox.originY),
      width: Math.round(bestFace.boundingBox.width),
      height: Math.round(bestFace.boundingBox.height),
    },
    faceCanvas.width,
    faceCanvas.height
  );

  state.faceDetected = true;
  state.faceConfidence = getFaceConfidence(bestFace);
  state.faceLaplacianVariance = computeFaceLaplacianVariance(bestFace.boundingBox);
  state.faceBlurry = state.faceLaplacianVariance < FACE_BLUR_LAPLACIAN_THRESHOLD;
  state.faceBoxInAnalysis = {
    x: guideRect.x + bestBox.x,
    y: guideRect.y + bestBox.y,
    width: bestBox.width,
    height: bestBox.height,
  };
}

function renderIssues(list) {
  issuesList.innerHTML = "";

  if (list.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No issues";
    issuesList.appendChild(li);
    return;
  }

  for (const issue of list) {
    const li = document.createElement("li");
    li.textContent = issue;
    issuesList.appendChild(li);
  }
}

function resetDetectionState() {
  state.idDetected = false;
  state.fitsGuide = false;
  state.blurred = false;
  state.outOfFocus = false;
  state.wholeIdUnclear = false;
  state.faceDetected = false;
  state.faceBlurry = false;
  state.faceConfidence = 0;
  state.faceBoxInAnalysis = null;
  state.lastGuideRect = null;
  state.issues = [ISSUE_NO_ID];
  state.laplacianVariance = 0;
  state.tenengrad = 0;
  state.faceLaplacianVariance = 0;
  timing.lastIdAt = 0;
  timing.lastFaceAt = 0;
  timing.lastUiAt = 0;
  uiStability.color = "white";
  uiStability.issues = [ISSUE_NO_ID];
  uiStability.candidateColor = "white";
  uiStability.candidateIssues = [ISSUE_NO_ID];
  uiStability.candidateSince = 0;
  clearFaceOverlay();
}

function getRawIssues() {
  if (!state.idDetected) {
    return [ISSUE_NO_ID];
  }

  const issues = [];
  if (!state.fitsGuide) {
    issues.push(ISSUE_ID_NOT_IN_FRAME);
  }
  if (state.wholeIdUnclear) {
    issues.push(ISSUE_WHOLE_ID_BLUR);
  }
  if (!state.faceDetected) {
    issues.push(ISSUE_NO_FACE);
  }
  if (state.faceDetected && state.faceBlurry) {
    issues.push(ISSUE_BLURRY_FACE);
  }
  return issues;
}

function getUiColorFromIssues(issues) {
  if (issues.length === 1 && issues[0] === ISSUE_NO_ID) {
    return "white";
  }
  return issues.length === 0 ? "green" : "red";
}

function getTransitionDelayMs(currentColor, nextColor) {
  const key = `${currentColor}To${nextColor.charAt(0).toUpperCase()}${nextColor.slice(1)}`;
  return STATE_TRANSITION_DELAY_MS[key] ?? 260;
}

function getStableUiDecision(nowMs) {
  const rawIssues = getRawIssues();
  const rawColor = getUiColorFromIssues(rawIssues);

  if (rawColor === uiStability.color) {
    uiStability.issues = rawIssues;
    uiStability.candidateColor = rawColor;
    uiStability.candidateIssues = rawIssues;
    uiStability.candidateSince = nowMs;
    return { color: uiStability.color, issues: uiStability.issues };
  }

  if (rawColor !== uiStability.candidateColor) {
    uiStability.candidateColor = rawColor;
    uiStability.candidateIssues = rawIssues;
    uiStability.candidateSince = nowMs;
  }

  const delayMs = getTransitionDelayMs(uiStability.color, uiStability.candidateColor);
  if (nowMs - uiStability.candidateSince >= delayMs) {
    uiStability.color = uiStability.candidateColor;
    uiStability.issues = uiStability.candidateIssues;
  }

  return { color: uiStability.color, issues: uiStability.issues };
}

function updateUi(nowMs = performance.now()) {
  guide.classList.remove("guide-white", "guide-red", "guide-green");

  if (scanLocked) {
    guide.classList.add("guide-green");
    statusLabel.textContent = "Photo submitted. Tap Redo Scan to scan again.";
    renderIssues([]);
    setButtonState(submitButton, false);
    setButtonState(redoButton, true);
    clearFaceOverlay();
    return;
  }

  const uiDecision = getStableUiDecision(nowMs);
  const { color, issues } = uiDecision;
  state.issues = issues;

  if (color === "white") {
    guide.classList.add("guide-white");
    statusLabel.textContent = "Show your ID in the camera.";
    renderIssues(issues);
    setButtonState(submitButton, false);
    setButtonState(redoButton, false);
    clearFaceOverlay();
    return;
  }

  if (color === "green") {
    guide.classList.add("guide-green");
    statusLabel.textContent = "All checks passed. You can submit the photo.";
    renderIssues([]);
    setButtonState(submitButton, true);
    setButtonState(redoButton, false);
    renderFaceOverlay();
    return;
  }

  guide.classList.add("guide-red");
  statusLabel.textContent = "Fix the issues below.";
  renderIssues(issues);
  setButtonState(submitButton, false);
  setButtonState(redoButton, false);
  renderFaceOverlay();
}

function analyzeFrame() {
  drawCurrentFrameToAnalysisCanvas();

  let src;
  let gray;

  try {
    src = cv.imread(analysisCanvas);
    gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    const guideRect = getGuideRectInAnalysisSpace();

    const idResult = detectIdRectangle(gray, guideRect);
    state.idDetected = idResult.detected;
    state.fitsGuide = idResult.fits;
    state.lastGuideRect = guideRect;

    if (state.idDetected) {
      const sharpness = computeSharpnessFromGrayRoi(gray, guideRect);
      state.blurred = sharpness.blurred;
      state.outOfFocus = sharpness.outOfFocus;
      state.wholeIdUnclear = sharpness.blurred || sharpness.outOfFocus;
      state.laplacianVariance = sharpness.lapVariance;
      state.tenengrad = sharpness.tenengrad;
    } else {
      state.blurred = false;
      state.outOfFocus = false;
      state.wholeIdUnclear = false;
      state.faceDetected = false;
      state.faceBlurry = false;
      state.faceConfidence = 0;
      state.faceBoxInAnalysis = null;
      state.lastGuideRect = null;
      state.faceLaplacianVariance = 0;
    }

    return guideRect;
  } finally {
    if (src) {
      src.delete();
    }
    if (gray) {
      gray.delete();
    }
  }
}

function processingLoop(now) {
  loopHandle = requestAnimationFrame(processingLoop);

  if (!cameraReady || !cvReady || scanLocked) {
    return;
  }

  let analyzedThisCycle = false;

  if (now - timing.lastIdAt >= ID_INTERVAL_MS) {
    timing.lastIdAt = now;
    analyzeFrame();
    analyzedThisCycle = true;
  }

  if (analyzedThisCycle && state.idDetected && state.lastGuideRect && now - timing.lastFaceAt >= FACE_INTERVAL_MS) {
    timing.lastFaceAt = now;
    updateFaceDetection(state.lastGuideRect);
  }

  if (now - timing.lastUiAt >= UI_INTERVAL_MS) {
    timing.lastUiAt = now;
    updateUi(now);
  }
}

function stopCamera() {
  cameraReady = false;

  if (video.srcObject instanceof MediaStream) {
    for (const track of video.srcObject.getTracks()) {
      track.stop();
    }
  }

  activeStream = null;
  video.pause();
  video.srcObject = null;
}

async function startCamera() {
  if (activeStream) {
    video.srcObject = activeStream;
    await video.play();
    cameraReady = true;
    return;
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: "environment" },
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  });

  activeStream = stream;
  video.srcObject = stream;

  await new Promise((resolve) => {
    if (video.readyState >= 1) {
      resolve();
      return;
    }

    video.onloadedmetadata = () => {
      resolve();
    };
  });

  await video.play();
  cameraReady = true;
}

function captureCurrentId() {
  if (submitButton.disabled || scanLocked) {
    return;
  }

  const renderWidth = video.clientWidth;
  const renderHeight = video.clientHeight;
  const crop = getVideoCoverCrop(video.videoWidth, video.videoHeight, renderWidth, renderHeight);

  const videoRect = video.getBoundingClientRect();
  const guideRect = guide.getBoundingClientRect();

  const rx = guideRect.left - videoRect.left;
  const ry = guideRect.top - videoRect.top;
  const rw = guideRect.width;
  const rh = guideRect.height;

  const sx = crop.sx + (rx / renderWidth) * crop.sw;
  const sy = crop.sy + (ry / renderHeight) * crop.sh;
  const sw = (rw / renderWidth) * crop.sw;
  const sh = (rh / renderHeight) * crop.sh;

  captureCanvas.width = Math.round(sw);
  captureCanvas.height = Math.round(sh);

  captureCtx.drawImage(video, sx, sy, sw, sh, 0, 0, captureCanvas.width, captureCanvas.height);

  const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.92);
  capturePreview.src = dataUrl;
  downloadLink.href = dataUrl;
  captureResult.classList.remove("hidden");

  scanLocked = true;
  stopCamera();
  updateUi();
}

async function redoScan() {
  if (!scanLocked) {
    return;
  }

  setButtonState(redoButton, false);
  statusLabel.textContent = "Restarting camera...";

  scanLocked = false;
  captureResult.classList.add("hidden");
  capturePreview.removeAttribute("src");
  downloadLink.removeAttribute("href");
  resetDetectionState();

  try {
    await startCamera();
    updateUi();
  } catch (error) {
    onCameraError(error);
  }
}

function onCameraError(error) {
  console.error(error);
  statusLabel.textContent = "Initialization failed.";
  renderIssues(["Allow camera access and ensure network can load model files."]);
  setButtonState(submitButton, false);
  setButtonState(redoButton, false);
  clearFaceOverlay();
}

async function setupFaceDetector() {
  const vision = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17");
  const fileset = await vision.FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm"
  );

  faceDetector = await vision.FaceDetector.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
      delegate: "GPU",
    },
    runningMode: "IMAGE",
    minDetectionConfidence: 0.35,
  });
}

function waitForOpenCv(maxMs = 25000) {
  return new Promise((resolve, reject) => {
    const started = performance.now();

    const check = () => {
      if (window.cv && cv.Mat) {
        resolve();
        return;
      }

      if (performance.now() - started > maxMs) {
        reject(new Error("OpenCV did not initialize in time."));
        return;
      }

      window.setTimeout(check, 120);
    };

    check();
  });
}

async function init() {
  if (!navigator.mediaDevices?.getUserMedia) {
    statusLabel.textContent = "Camera not supported.";
    renderIssues(["Use a modern mobile browser (Chrome or Samsung Internet)."]);
    return;
  }

  submitButton.addEventListener("click", captureCurrentId);
  redoButton.addEventListener("click", () => {
    redoScan();
  });

  try {
    await Promise.all([startCamera(), waitForOpenCv(), setupFaceDetector()]);
    cvReady = true;
    statusLabel.textContent = "Camera ready. Align your ID inside the rectangle.";
    renderIssues([ISSUE_NO_ID]);
    updateUi();
    loopHandle = requestAnimationFrame(processingLoop);
  } catch (error) {
    onCameraError(error);
  }
}

window.addEventListener("beforeunload", () => {
  if (loopHandle) {
    cancelAnimationFrame(loopHandle);
  }

  stopCamera();
});

init();
