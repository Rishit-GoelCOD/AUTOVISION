"""
src/realtime_monitor.py — Real-Time Driver Safety Monitor

College ML Project: MediaPipe Face Mesh → Geometric Features → StandardScaler → Logistic Regression.

Architecture:
  1. YAWN:         MAR       → yawn_scaler.joblib         → yawn_model.joblib         → Yawn Prob
  2. EYES CLOSED:  EAR       → eyes_closed_scaler.joblib  → eyes_closed_model.joblib  → Eyes Closed Prob
  3. LOOKING AWAY: abs(yaw)  → looking_away_scaler.joblib → looking_away_model.joblib → Looking Away Prob
  4. LOOKING DOWN: pitch     → looking_down_scaler.joblib → looking_down_model.joblib → Looking Down Prob

Temporal Stability:
  Requires conditions to persist for consecutive frames before triggering an active warning.
"""

import sys
import time
import subprocess
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & Imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent
MODELS_DIR = BASE_DIR / "models"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from features import extract_features

# ---------------------------------------------------------------------------
# Model file definitions
# ---------------------------------------------------------------------------
MODEL_SPECS = {
    "yawn": {
        "model_path":  MODELS_DIR / "yawn_model.joblib",
        "scaler_path": MODELS_DIR / "yawn_scaler.joblib",
        "feature_col": "MAR",
        "label":       "YAWNING",
        "normal_text": "NORMAL",
        "alert_text":  "YAWNING",
    },
    "eyes_closed": {
        "model_path":  MODELS_DIR / "eyes_closed_model.joblib",
        "scaler_path": MODELS_DIR / "eyes_closed_scaler.joblib",
        "feature_col": "EAR",
        "label":       "EYES",
        "normal_text": "OPEN",
        "alert_text":  "CLOSED",
    },
    "looking_away": {
        "model_path":  MODELS_DIR / "looking_away_model.joblib",
        "scaler_path": MODELS_DIR / "looking_away_scaler.joblib",
        "feature_col": "yaw",  # trained on abs(yaw) saved as column 'yaw'
        "label":       "LOOKING AWAY",
        "normal_text": "NO",
        "alert_text":  "YES",
    },
    "looking_down": {
        "model_path":  MODELS_DIR / "looking_down_model.joblib",
        "scaler_path": MODELS_DIR / "looking_down_scaler.joblib",
        "feature_col": "pitch",
        "label":       "LOOKING DOWN",
        "normal_text": "NO",
        "alert_text":  "YES",
    },
}

# ---------------------------------------------------------------------------
# Thresholds & Consecutive Frame Requirements
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "yawn":         0.50,
    "eyes_closed":  0.50,
    "looking_away": 0.60,  # Increased bandwidth: requires ~13-14% deviation before triggering
    "looking_down": 0.50,
}

# Frame persistence (~30 fps camera)
CONSECUTIVE_FRAMES = {
    "yawn":         10,
    "eyes_closed":  10,
    "looking_away": 12,    # ~0.4s sustained turn: ignores brief glances/micro-movements
    "looking_down": 8,
}

# Sound alert throttle
ALERT_SOUND_PATH = Path("/System/Library/Sounds/Tink.aiff")
AUDIO_COOLDOWN_SEC = 2.0

# ---------------------------------------------------------------------------
# Colors (BGR)
# ---------------------------------------------------------------------------
COLOR_BG       = (20, 22, 28)
COLOR_TEXT     = (240, 240, 245)
COLOR_MUTED    = (160, 165, 175)
COLOR_GREEN    = (60, 210, 80)
COLOR_ORANGE   = (0, 150, 255)
COLOR_RED      = (45, 50, 230)
COLOR_YELLOW   = (0, 215, 255)
COLOR_CYAN     = (255, 205, 50)
COLOR_BORDER   = (65, 75, 95)


# ===========================================================================
# 1. Load Trained Models & Scalers
# ===========================================================================
def load_all_models() -> dict:
    """Loads all 4 Logistic Regression models and StandardScalers."""
    loaded = {}
    print("=" * 60)
    print("  Loading Logistic Regression Models & Scalers...")
    print("=" * 60)

    for key, spec in MODEL_SPECS.items():
        m_path = spec["model_path"]
        s_path = spec["scaler_path"]

        if not m_path.exists() or not s_path.exists():
            print(f"[ERROR] Missing model or scaler file for '{key}':")
            print(f"  Model:  {m_path} (exists={m_path.exists()})")
            print(f"  Scaler: {s_path} (exists={s_path.exists()})")
            print("Please run 'python src/train_models.py' first.")
            sys.exit(1)

        model = joblib.load(m_path)
        scaler = joblib.load(s_path)
        loaded[key] = (model, scaler)
        print(f"  ✓ {key:<14}: Model={m_path.name}, Scaler={s_path.name}")

    print("=" * 60)
    print("  All 4 models loaded successfully.\n")
    return loaded


# ===========================================================================
# 2. Real-Time Inference
# ===========================================================================
def predict_driver_probabilities(loaded_models: dict,
                                 ear: float,
                                 mar: float,
                                 yaw: float,
                                 pitch: float) -> dict[str, float]:
    """
    Computes probabilities for all four conditions.
    IMPORTANT: abs(yaw) is computed before scaling and predicting looking away.
    Uses named pandas DataFrames to prevent feature-name warnings.
    """
    yawn_m,  yawn_s  = loaded_models["yawn"]
    eyes_m,  eyes_s  = loaded_models["eyes_closed"]
    look_m,  look_s  = loaded_models["looking_away"]
    down_m,  down_s  = loaded_models["looking_down"]

    # 1. Yawn (MAR)
    mar_df = pd.DataFrame([[mar]], columns=["MAR"])
    mar_scaled = yawn_s.transform(mar_df)
    p_yawn = float(yawn_m.predict_proba(mar_scaled)[0][1])

    # 2. Eyes Closed (EAR)
    ear_df = pd.DataFrame([[ear]], columns=["EAR"])
    ear_scaled = eyes_s.transform(ear_df)
    p_eyes = float(eyes_m.predict_proba(ear_scaled)[0][1])

    # 3. Looking Away (abs(yaw) - measures deviation from center)
    abs_yaw = abs(yaw)
    yaw_df = pd.DataFrame([[abs_yaw]], columns=["yaw"])
    yaw_scaled = look_s.transform(yaw_df)
    p_away = float(look_m.predict_proba(yaw_scaled)[0][1])

    # 4. Looking Down (pitch)
    pitch_df = pd.DataFrame([[pitch]], columns=["pitch"])
    pitch_scaled = down_s.transform(pitch_df)
    p_down = float(down_m.predict_proba(pitch_scaled)[0][1])

    return {
        "yawn":         p_yawn,
        "eyes_closed":  p_eyes,
        "looking_away": p_away,
        "looking_down": p_down,
    }


# ===========================================================================
# 3. Audio Alert Helper
# ===========================================================================
_last_sound_time = 0.0

def play_alert_sound():
    """Plays an alert chime on macOS without blocking."""
    global _last_sound_time
    now = time.time()
    if now - _last_sound_time < AUDIO_COOLDOWN_SEC:
        return
    _last_sound_time = now

    if sys.platform == "darwin" and ALERT_SOUND_PATH.exists():
        try:
            subprocess.Popen(
                ["afplay", str(ALERT_SOUND_PATH)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


# ===========================================================================
# 4. HUD Drawing Functions
# ===========================================================================
def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=8):
    """Draws a clean rectangle with optional rounded aesthetics."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def draw_hud(frame: np.ndarray,
             face_detected: bool,
             ear: float | None,
             mar: float | None,
             yaw: float | None,
             pitch: float | None,
             probs: dict[str, float],
             counters: dict[str, int],
             active_alerts: list[str],
             fps: float,
             show_raw_debug: bool = True) -> np.ndarray:
    """Renders the Driver Safety Monitor HUD overlay."""
    h, w = frame.shape[:2]

    # --- Top Banner Area ---
    banner_h = 55
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (12, 14, 18), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Title
    cv2.putText(frame, "DRIVER SAFETY MONITOR", (20, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, COLOR_TEXT, 2, cv2.LINE_AA)

    # FPS Indicator
    fps_text = f"FPS: {fps:4.1f}"
    cv2.putText(frame, fps_text, (w - 120, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLOR_MUTED, 1, cv2.LINE_AA)

    # --- Main Status Banner (below top title) ---
    status_y = banner_h + 10
    status_h = 42

    if not face_detected:
        status_bg = (0, 120, 210)  # Orange
        status_text = "⚠  FACE NOT DETECTED"
        text_color = (255, 255, 255)
    elif len(active_alerts) == 0:
        status_bg = (35, 130, 45)   # Green
        status_text = "●  STATUS: NORMAL (DRIVER ATTENTIVE)"
        text_color = (245, 255, 245)
    else:
        status_bg = (30, 35, 190)   # Red
        alert_str = " | ".join(active_alerts)
        status_text = f"▲  WARNING: {alert_str}"
        text_color = (255, 255, 255)

    status_overlay = frame.copy()
    cv2.rectangle(status_overlay, (15, status_y), (w - 15, status_y + status_h), status_bg, -1)
    cv2.addWeighted(status_overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (15, status_y), (w - 15, status_y + status_h), COLOR_BORDER, 1)

    cv2.putText(frame, status_text, (30, status_y + 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, text_color, 2, cv2.LINE_AA)

    # --- Left HUD Panel: 4 Driver-State Indicators ---
    panel_x1, panel_y1 = 15, status_y + status_h + 12
    panel_w = 410
    panel_h = 240 if show_raw_debug else 170
    panel_x2, panel_y2 = panel_x1 + panel_w, panel_y1 + panel_h

    panel_overlay = frame.copy()
    cv2.rectangle(panel_overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (14, 16, 22), -1)
    cv2.addWeighted(panel_overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), COLOR_BORDER, 1)

    # Table Header
    table_y = panel_y1 + 26
    cv2.putText(frame, "CONDITION", (panel_x1 + 15, table_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_YELLOW, 1, cv2.LINE_AA)
    cv2.putText(frame, "STATE", (panel_x1 + 175, table_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_YELLOW, 1, cv2.LINE_AA)
    cv2.putText(frame, "PROB", (panel_x1 + 270, table_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_YELLOW, 1, cv2.LINE_AA)
    cv2.putText(frame, "FRAMES", (panel_x1 + 335, table_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_YELLOW, 1, cv2.LINE_AA)

    cv2.line(frame, (panel_x1 + 12, table_y + 8), (panel_x2 - 12, table_y + 8), COLOR_BORDER, 1)

    row_keys = ["yawn", "eyes_closed", "looking_away", "looking_down"]
    row_y_start = table_y + 32
    row_spacing = 30

    for idx, key in enumerate(row_keys):
        spec = MODEL_SPECS[key]
        ry = row_y_start + idx * row_spacing

        p = probs.get(key, 0.0)
        c = counters.get(key, 0)
        max_c = CONSECUTIVE_FRAMES[key]
        is_alert = (c >= max_c)

        # Condition name
        cv2.putText(frame, spec["label"], (panel_x1 + 15, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_TEXT, 1, cv2.LINE_AA)

        # State text & color
        if is_alert:
            state_str = spec["alert_text"]
            state_color = COLOR_RED
        elif p >= THRESHOLDS[key]:
            state_str = f"PENDING"
            state_color = COLOR_ORANGE
        else:
            state_str = spec["normal_text"]
            state_color = COLOR_GREEN

        cv2.putText(frame, state_str, (panel_x1 + 175, ry),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, state_color, 1, cv2.LINE_AA)

        # Probability
        prob_str = f"{p:.2f}"
        cv2.putText(frame, prob_str, (panel_x1 + 270, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, state_color, 1, cv2.LINE_AA)

        # Counter / frames
        cnt_str = f"{min(c, max_c):2d}/{max_c}"
        cv2.putText(frame, cnt_str, (panel_x1 + 340, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_MUTED, 1, cv2.LINE_AA)

    # Optional Raw Features sub-panel
    if show_raw_debug and ear is not None:
        raw_y = row_y_start + 4 * row_spacing + 4
        cv2.line(frame, (panel_x1 + 12, raw_y - 12), (panel_x2 - 12, raw_y - 12), COLOR_BORDER, 1)
        raw_str1 = f"EAR: {ear:.3f}   MAR: {mar:.3f}"
        raw_str2 = f"Yaw: {yaw:+.1f}% (|yaw|: {abs(yaw):.1f}%)   Pitch: {pitch:+.1f}%"
        cv2.putText(frame, raw_str1, (panel_x1 + 15, raw_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_CYAN, 1, cv2.LINE_AA)
        cv2.putText(frame, raw_str2, (panel_x1 + 15, raw_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_CYAN, 1, cv2.LINE_AA)

    # --- Right Side: Probability Gauge Bars ---
    bar_panel_w = 140
    bar_panel_x1 = w - bar_panel_w - 15
    bar_panel_y1 = panel_y1
    bar_panel_x2 = w - 15
    bar_panel_y2 = panel_y2

    bar_overlay = frame.copy()
    cv2.rectangle(bar_overlay, (bar_panel_x1, bar_panel_y1), (bar_panel_x2, bar_panel_y2), (14, 16, 22), -1)
    cv2.addWeighted(bar_overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (bar_panel_x1, bar_panel_y1), (bar_panel_x2, bar_panel_y2), COLOR_BORDER, 1)

    cv2.putText(frame, "PROBABILITIES", (bar_panel_x1 + 15, bar_panel_y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_YELLOW, 1, cv2.LINE_AA)

    # Draw 4 vertical bars
    bar_w = 18
    bar_max_h = panel_h - 60
    bar_gap = 12
    total_bars_w = 4 * bar_w + 3 * bar_gap
    start_bx = bar_panel_x1 + (bar_panel_w - total_bars_w) // 2

    bar_labels = ["Ywn", "Eye", "Awy", "Dwn"]
    for i, key in enumerate(row_keys):
        bx = start_bx + i * (bar_w + bar_gap)
        by = bar_panel_y1 + 40
        p = probs.get(key, 0.0)

        # Bar background
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_max_h), (35, 38, 48), -1)

        # Bar fill
        fill_h = int(bar_max_h * max(0.0, min(1.0, p)))
        fill_color = COLOR_RED if p >= 0.70 else (COLOR_ORANGE if p >= 0.50 else COLOR_GREEN)
        cv2.rectangle(frame, (bx, by + bar_max_h - fill_h), (bx + bar_w, by + bar_max_h), fill_color, -1)

        # 50% Threshold marker
        thresh_y = by + bar_max_h - int(bar_max_h * 0.50)
        cv2.line(frame, (bx - 2, thresh_y), (bx + bar_w + 2, thresh_y), COLOR_YELLOW, 1)

        # Label under bar
        cv2.putText(frame, bar_labels[i], (bx - 2, by + bar_max_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_MUTED, 1, cv2.LINE_AA)

    # --- Bottom Footer Bar ---
    footer_h = 30
    footer_overlay = frame.copy()
    cv2.rectangle(footer_overlay, (0, h - footer_h), (w, h), (12, 14, 18), -1)
    cv2.addWeighted(footer_overlay, 0.85, frame, 0.15, 0, frame)
    footer_text = "Press 'q' to Quit  |  Press 'd' to Toggle Raw Biomarkers  |  Concept: Camera Mounted Below Looking Up"
    cv2.putText(frame, footer_text, (20, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_MUTED, 1, cv2.LINE_AA)

    return frame


# ===========================================================================
# 5. Main Real-Time Loop
# ===========================================================================
def main() -> None:
    print("=" * 65)
    print("      DRIVER SAFETY MONITOR — REAL-TIME MONITORING")
    print("=" * 65)
    print("  Controls:")
    print("    'q' : Quit application cleanly")
    print("    'd' : Toggle raw facial features on HUD\n")

    # Load 4 models & scalers
    loaded_models = load_all_models()

    # Open webcam
    print("Opening webcam device (index 0)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam. Verify permissions and connection.")
        sys.exit(1)

    # Request smooth frame rate and HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # State counters for temporal smoothing
    counters = {
        "yawn":         0,
        "eyes_closed":  0,
        "looking_away": 0,
        "looking_down": 0,
    }
    no_face_frames = 0

    # Rolling FPS measurement
    prev_time = time.time()
    fps = 30.0
    show_raw_debug = True

    print("Webcam active. Starting Driver Safety Monitor window...\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # ── 1. Feature Extraction ─────────────────────────────────────────
            features = extract_features(frame)

            if features is None:
                no_face_frames += 1
                # Smoothly decay active state counters
                for k in counters:
                    counters[k] = max(0, counters[k] - 1)

                active_alerts = []
                # Only if face is absent for a sustained period (~0.6s) do we alert
                if no_face_frames >= 18:
                    active_alerts.append("NO FACE / LOOKING AWAY")
                    play_alert_sound()

                frame = draw_hud(
                    frame          = frame,
                    face_detected  = False,
                    ear            = None,
                    mar            = None,
                    yaw            = None,
                    pitch          = None,
                    probs          = {
                        "yawn": 0.0,
                        "eyes_closed": 0.0,
                        "looking_away": 1.0 if no_face_frames >= 18 else 0.0,
                        "looking_down": 0.0,
                    },
                    counters       = counters,
                    active_alerts  = active_alerts,
                    fps            = fps,
                    show_raw_debug = show_raw_debug,
                )

            else:
                no_face_frames = 0
                ear, mar, yaw, pitch = features

                # ── 2. Real-Time Model Inference ──────────────────────────────
                probs = predict_driver_probabilities(
                    loaded_models, ear, mar, yaw, pitch
                )

                # ── 3. Temporal Counter Updates ───────────────────────────────
                for k, p in probs.items():
                    if p >= THRESHOLDS[k]:
                        counters[k] += 1
                    else:
                        counters[k] = max(0, counters[k] - 2)  # fast decay when condition ends

                # ── 4. Active Alert Determination ─────────────────────────────
                active_alerts = []
                if counters["eyes_closed"] >= CONSECUTIVE_FRAMES["eyes_closed"]:
                    active_alerts.append("EYES CLOSED")
                if counters["looking_away"] >= CONSECUTIVE_FRAMES["looking_away"]:
                    active_alerts.append("LOOKING AWAY")
                if counters["looking_down"] >= CONSECUTIVE_FRAMES["looking_down"]:
                    active_alerts.append("LOOKING DOWN")
                if counters["yawn"] >= CONSECUTIVE_FRAMES["yawn"]:
                    active_alerts.append("YAWNING")

                # Play gentle audio chime if an alert is active
                if len(active_alerts) > 0:
                    play_alert_sound()

                # ── 5. Render HUD ─────────────────────────────────────────────
                frame = draw_hud(
                    frame          = frame,
                    face_detected  = True,
                    ear            = ear,
                    mar            = mar,
                    yaw            = yaw,
                    pitch          = pitch,
                    probs          = probs,
                    counters       = counters,
                    active_alerts  = active_alerts,
                    fps            = fps,
                    show_raw_debug = show_raw_debug,
                )

            # Display window
            cv2.imshow("AUTOVISION — Real-Time Driver Safety Monitor", frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[INFO] 'q' pressed. Exiting monitor cleanly...")
                break
            elif key == ord("d"):
                show_raw_debug = not show_raw_debug

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Exiting...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released and display windows closed.")
        print("Driver Safety Monitor shutdown complete.")


if __name__ == "__main__":
    main()
