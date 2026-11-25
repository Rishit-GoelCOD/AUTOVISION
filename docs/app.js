/**
 * AUTOVISION — Client-Side Driver Safety Monitor (WebAssembly & JS)
 * Uses MediaPipe Face Mesh to extract geometric landmarks, computes
 * invariant biomarkers (EAR, MAR, |Yaw|, Pitch), and performs real-time classification.
 */

// Landmark Index Constants
const LEFT_EYE = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];
const MOUTH = { top: 13, bottom: 14, left: 78, right: 308 };
const NOSE_TIP = 1;
const CHIN = 152;
const CHEEK_LEFT = 234;
const CHEEK_RIGHT = 454;

// Thresholds & Persistence Limits
const PERSISTENCE_LIMITS = {
  eyes_closed: 8,
  yawn: 12,
  looking_away: 10,
  looking_down: 8,
  no_face: 15,
};

// State Variables
let faceMesh = null;
let camera = null;
let isRunning = false;
let audioCtx = null;
let lastAlertSoundTime = 0;

let counters = {
  eyes_closed: 0,
  yawn: 0,
  looking_away: 0,
  looking_down: 0,
};
let noFaceFrames = 0;

// Rolling FPS calculation
let lastFrameTime = performance.now();
let fps = 30.0;

// DOM Elements
const videoElement = document.getElementById("webcam-video");
const canvasElement = document.getElementById("output-canvas");
const canvasCtx = canvasElement.getContext("2d");
const startBtn = document.getElementById("btn-start");
const stopBtn = document.getElementById("btn-stop");
const fpsDisplay = document.getElementById("fps-display");
const alertBanner = document.getElementById("alert-banner");

// Telemetry Elements
const earValueEl = document.getElementById("val-ear");
const earFillEl = document.getElementById("fill-ear");
const marValueEl = document.getElementById("val-mar");
const marFillEl = document.getElementById("fill-mar");
const yawValueEl = document.getElementById("val-yaw");
const yawFillEl = document.getElementById("fill-yaw");
const pitchValueEl = document.getElementById("val-pitch");
const pitchFillEl = document.getElementById("fill-pitch");
const overallStatusEl = document.getElementById("overall-status");
const statusValueEl = document.getElementById("status-value");

// Web Audio API Chime Synthesizer
function playAlertChime() {
  const now = Date.now();
  if (now - lastAlertSoundTime < 1400) return; // rate limit chime
  lastAlertSoundTime = now;

  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === "suspended") {
      audioCtx.resume();
    }
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    osc.type = "sine";
    osc.frequency.setValueAtTime(880, audioCtx.currentTime); // A5
    osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.35);

    gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);

    osc.connect(gain);
    gain.connect(audioCtx.destination);

    osc.start();
    osc.stop(audioCtx.currentTime + 0.35);
  } catch (e) {
    console.warn("Audio chime failed to play:", e);
  }
}

// Distance helper
function euclideanDist(p1, p2) {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

// Sigmoid function
function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

// Calculate Eye Aspect Ratio
function calcEyeEAR(eyePoints, landmarks) {
  const p1 = landmarks[eyePoints[0]];
  const p2 = landmarks[eyePoints[1]];
  const p3 = landmarks[eyePoints[2]];
  const p4 = landmarks[eyePoints[3]];
  const p5 = landmarks[eyePoints[4]];
  const p6 = landmarks[eyePoints[5]];

  const vertical1 = euclideanDist(p2, p6);
  const vertical2 = euclideanDist(p3, p5);
  const horizontal = euclideanDist(p1, p4);

  if (horizontal === 0) return 0.3;
  return (vertical1 + vertical2) / (2.0 * horizontal);
}

// FaceMesh Results Callback
function onResults(results) {
  // Update FPS
  const now = performance.now();
  const dt = (now - lastFrameTime) / 1000.0;
  lastFrameTime = now;
  if (dt > 0) {
    fps = 0.85 * fps + 0.15 * (1.0 / dt);
  }
  fpsDisplay.textContent = `FPS: ${Math.round(fps)}`;

  // Set canvas size matching video
  if (canvasElement.width !== videoElement.videoWidth || canvasElement.height !== videoElement.videoHeight) {
    canvasElement.width = videoElement.videoWidth || 640;
    canvasElement.height = videoElement.videoHeight || 480;
  }

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  let activeAlerts = [];

  if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
    noFaceFrames = 0;
    const landmarks = results.multiFaceLandmarks[0];

    // Optional: Draw subtle landmark mesh points
    canvasCtx.fillStyle = "rgba(0, 210, 255, 0.4)";
    for (let i = 0; i < landmarks.length; i += 8) {
      const p = landmarks[i];
      canvasCtx.beginPath();
      canvasCtx.arc(p.x * canvasElement.width, p.y * canvasElement.height, 1.5, 0, 2 * Math.PI);
      canvasCtx.fill();
    }

    // 1. EAR (Eye Aspect Ratio)
    const leftEAR = calcEyeEAR(LEFT_EYE, landmarks);
    const rightEAR = calcEyeEAR(RIGHT_EYE, landmarks);
    const ear = (leftEAR + rightEAR) / 2.0;

    // 2. MAR (Mouth Aspect Ratio)
    const mouthTop = landmarks[MOUTH.top];
    const mouthBottom = landmarks[MOUTH.bottom];
    const mouthLeft = landmarks[MOUTH.left];
    const mouthRight = landmarks[MOUTH.right];
    const marVert = euclideanDist(mouthTop, mouthBottom);
    const marHoriz = euclideanDist(mouthLeft, mouthRight);
    const mar = marHoriz > 0 ? marVert / marHoriz : 0.2;

    // 3. Absolute Yaw (Horizontal Nose Symmetry)
    const nose = landmarks[NOSE_TIP];
    const cheekLeft = landmarks[CHEEK_LEFT];
    const cheekRight = landmarks[CHEEK_RIGHT];
    const dLeft = euclideanDist(nose, cheekLeft);
    const dRight = euclideanDist(nose, cheekRight);
    const yaw = dRight > 0 ? Math.abs(Math.log10(dLeft / dRight)) : 0.0;

    // 4. Pitch (Vertical Nose Proportion)
    const eyeMidY = (landmarks[LEFT_EYE[0]].y + landmarks[RIGHT_EYE[3]].y) / 2.0;
    const chinY = landmarks[CHIN].y;
    const pitch = (chinY - eyeMidY) !== 0 ? (nose.y - eyeMidY) / (chinY - eyeMidY) : 0.35;

    // Machine Learning Probabilities (Calibrated Sigmoid Models)
    const pEyesClosed = sigmoid((0.21 - ear) * 35.0);
    const pYawn = sigmoid((mar - 0.50) * 18.0);
    const pLookingAway = sigmoid((yaw - 0.22) * 22.0);
    const pLookingDown = sigmoid((pitch - 0.48) * 22.0);

    // Persistence Counters
    if (pEyesClosed > 0.50) counters.eyes_closed++;
    else counters.eyes_closed = Math.max(0, counters.eyes_closed - 1);

    if (pYawn > 0.50) counters.yawn++;
    else counters.yawn = Math.max(0, counters.yawn - 1);

    if (pLookingAway > 0.60) counters.looking_away++;
    else counters.looking_away = Math.max(0, counters.looking_away - 1);

    if (pLookingDown > 0.50) counters.looking_down++;
    else counters.looking_down = Math.max(0, counters.looking_down - 1);

    // Active alert checks
    if (counters.eyes_closed >= PERSISTENCE_LIMITS.eyes_closed) {
      activeAlerts.push("DROWSY (EYES CLOSED)");
    }
    if (counters.looking_down >= PERSISTENCE_LIMITS.looking_down) {
      activeAlerts.push("LOOKING DOWN (PHONE)");
    }
    if (counters.looking_away >= PERSISTENCE_LIMITS.looking_away) {
      activeAlerts.push("LOOKING AWAY");
    }
    if (counters.yawn >= PERSISTENCE_LIMITS.yawn) {
      activeAlerts.push("YAWNING");
    }

    // Update Telemetry UI
    earValueEl.textContent = `${ear.toFixed(3)} (${Math.round(pEyesClosed * 100)}%)`;
    earFillEl.style.width = `${Math.min(100, Math.max(0, pEyesClosed * 100))}%`;
    earFillEl.style.backgroundColor = pEyesClosed > 0.5 ? "var(--alert-danger)" : "var(--alert-safe)";

    marValueEl.textContent = `${mar.toFixed(3)} (${Math.round(pYawn * 100)}%)`;
    marFillEl.style.width = `${Math.min(100, Math.max(0, pYawn * 100))}%`;
    marFillEl.style.backgroundColor = pYawn > 0.5 ? "var(--alert-warning)" : "var(--alert-safe)";

    yawValueEl.textContent = `${yaw.toFixed(3)} (${Math.round(pLookingAway * 100)}%)`;
    yawFillEl.style.width = `${Math.min(100, Math.max(0, pLookingAway * 100))}%`;
    yawFillEl.style.backgroundColor = pLookingAway > 0.6 ? "var(--alert-warning)" : "var(--alert-safe)";

    pitchValueEl.textContent = `${pitch.toFixed(3)} (${Math.round(pLookingDown * 100)}%)`;
    pitchFillEl.style.width = `${Math.min(100, Math.max(0, pLookingDown * 100))}%`;
    pitchFillEl.style.backgroundColor = pLookingDown > 0.5 ? "var(--alert-danger)" : "var(--alert-safe)";
  } else {
    noFaceFrames++;
    for (let k in counters) {
      counters[k] = Math.max(0, counters[k] - 1);
    }
    if (noFaceFrames >= PERSISTENCE_LIMITS.no_face) {
      activeAlerts.push("NO FACE / LOOKING AWAY");
    }
  }

  // Handle Overall Alerts & Sound
  if (activeAlerts.length > 0) {
    playAlertChime();
    alertBanner.style.display = "block";
    alertBanner.textContent = `⚠️ ALERT: ${activeAlerts.join(" + ")}`;

    overallStatusEl.className = "overall-status danger";
    statusValueEl.textContent = activeAlerts[0];
  } else {
    alertBanner.style.display = "none";
    overallStatusEl.className = "overall-status";
    statusValueEl.textContent = "ATTENTIVE & SAFE";
  }

  canvasCtx.restore();
}

// Start Monitoring
async function startMonitor() {
  if (isRunning) return;

  startBtn.disabled = true;
  startBtn.textContent = "Loading MediaPipe...";

  try {
    if (!faceMesh) {
      faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      });

      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      faceMesh.onResults(onResults);
    }

    camera = new Camera(videoElement, {
      onFrame: async () => {
        if (isRunning) {
          await faceMesh.send({ image: videoElement });
        }
      },
      width: 640,
      height: 480,
    });

    await camera.start();
    isRunning = true;
    startBtn.style.display = "none";
    stopBtn.style.display = "inline-flex";
  } catch (err) {
    console.error("Camera access failed:", err);
    alert(`Could not start camera: ${err.message}. Please check webcam permissions.`);
    startBtn.disabled = false;
    startBtn.textContent = "▶ Start Live Camera";
  }
}

// Stop Monitoring
function stopMonitor() {
  if (!isRunning) return;
  isRunning = false;

  if (camera) {
    camera.stop();
  }

  startBtn.disabled = false;
  startBtn.style.display = "inline-flex";
  startBtn.textContent = "▶ Start Live Camera";
  stopBtn.style.display = "none";
  alertBanner.style.display = "none";

  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  overallStatusEl.className = "overall-status";
  statusValueEl.textContent = "STANDBY";
}

// Event Listeners
startBtn.addEventListener("click", startMonitor);
stopBtn.addEventListener("click", stopMonitor);
