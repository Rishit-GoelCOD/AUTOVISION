"""
Live Webcam Feature Testing Module for Driver Safety Monitoring.

Opens the webcam, applies MediaPipe Face Mesh feature extraction in real-time,
and renders an on-screen HUD overlay showing:
- EAR (Eye Aspect Ratio)
- MAR (Mouth Aspect Ratio)
- Head Pose: Yaw and Pitch (in degrees)
- Warning badge when no face is detected.

Also prints Yaw and Pitch to the terminal once per second for easy reading.
Press 'q' to quit the live feed.
"""

import sys
import time
from pathlib import Path

import cv2

# Ensure src/ is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from features import extract_features


def draw_hud(frame, features, fps=None):
    """
    Renders a semi-transparent HUD overlay with metrics on the frame.
    """
    h, w = frame.shape[:2]

    # Create semi-transparent overlay card on the top-left
    card_w, card_h = 320, 180 if features else 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (15 + card_w, 15 + card_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Card border
    cv2.rectangle(frame, (15, 15), (15 + card_w, 15 + card_h), (80, 80, 80), 1)

    # Header title
    title = "DRIVER SAFETY MONITOR"
    cv2.putText(frame, title, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2, cv2.LINE_AA)

    if features is None:
        # No face detected indicator
        cv2.putText(
            frame,
            "No face",
            (25, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        if fps is not None:
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (25, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
        return frame

    ear, mar, yaw, pitch = features

    # Metric colors based on general thresholds
    # EAR: green if >= 0.22, orange/red if < 0.22 (eyes closing)
    ear_color = (0, 255, 0) if ear >= 0.22 else (0, 80, 255)
    # MAR: green if < 0.50, orange/red if >= 0.50 (yawning)
    mar_color = (0, 255, 0) if mar < 0.50 else (0, 80, 255)
    # Yaw / Pitch: green if within normal forward range (+-15 deg), orange otherwise
    pose_color = (0, 255, 0) if (abs(yaw) <= 15 and abs(pitch) <= 15) else (0, 80, 255)

    lines = [
        (f"EAR   : {ear:.3f}", ear_color),
        (f"MAR   : {mar:.3f}", mar_color),
        (f"Yaw   : {yaw:+.1f} deg", pose_color),
        (f"Pitch : {pitch:+.1f} deg", pose_color),
    ]

    y_pos = 70
    for text, color in lines:
        cv2.putText(frame, text, (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y_pos += 24

    # Footer with FPS and quit instruction
    footer_text = f"FPS: {fps:.1f} | Press 'q' to quit" if fps else "Press 'q' to quit"
    cv2.putText(frame, footer_text, (25, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


def run_live_test(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera at index {camera_index}. Check camera connection and macOS permissions.")
        return

    window_name = "Driver Safety Monitor - Live Feature Test"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.time()
    last_terminal_print = time.time()
    fps = 0.0

    print("Camera started. Press 'q' on the preview window to exit.")
    print("-" * 55)
    print("Printing live Head Pose to terminal once per second:")
    print(f"{'Timestamp':<10} | {'Yaw (deg)':>12} | {'Pitch (deg)':>12} | Status")
    print("-" * 55)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame from camera.")
            break

        # Compute rolling FPS
        curr_time = time.time()
        time_diff = curr_time - prev_time
        if time_diff > 0:
            fps = 1.0 / time_diff
        prev_time = curr_time

        # Extract EAR, MAR, yaw, pitch or None
        features = extract_features(frame)

        # Print yaw and pitch to terminal once every 1.0 second
        if curr_time - last_terminal_print >= 1.0:
            last_terminal_print = curr_time
            time_str = time.strftime("%H:%M:%S")
            if features is not None:
                ear, mar, yaw, pitch = features
                status_notes = []
                if abs(yaw) > 15:
                    status_notes.append("Turned " + ("Right" if yaw > 0 else "Left"))
                if abs(pitch) > 15:
                    status_notes.append("Looking " + ("Up" if pitch > 0 else "Down"))
                if not status_notes:
                    status_notes.append("Forward (Normal)")
                note_str = ", ".join(status_notes)
                print(f"[{time_str}] | {yaw:>+10.1f} deg | {pitch:>+10.1f} deg | {note_str}")
            else:
                print(f"[{time_str}] | {'--':>10}     | {'--':>10}     | No face detected")

        # Draw overlay on frame
        draw_hud(frame, features, fps=fps)

        cv2.imshow(window_name, frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("-" * 55)
            print("Exiting live feature test...")
            break

    cap.release()
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)


def main():
    run_live_test(camera_index=0)


if __name__ == "__main__":
    main()
