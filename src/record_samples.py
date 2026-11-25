"""
Custom Dataset Recording Tool for Driver Safety Monitoring.

Opens the webcam feed and saves frames to `data/own/<label>/` while a key is held:
- [o] : eyes_open
- [c] : eyes_closed
- [f] : facing_forward
- [a] : looking_away
- [d] : eyes_open_looking_down
- [q] : Quit

Filenames include a unique session ID timestamp.
Displays real-time saved counts per label on-screen.
"""

from datetime import datetime
from pathlib import Path
import sys
import time

import cv2

# Map keys to label folders
KEY_LABEL_MAP = {
    ord("o"): "eyes_open",
    ord("c"): "eyes_closed",
    ord("f"): "facing_forward",
    ord("a"): "looking_away",
    ord("d"): "eyes_open_looking_down",
}

LABEL_KEY_DISPLAY = {
    "eyes_open": "o",
    "eyes_closed": "c",
    "facing_forward": "f",
    "looking_away": "a",
    "eyes_open_looking_down": "d",
}


def get_base_dir() -> Path:
    """Finds the driver-safety-monitor root directory."""
    script_parent = Path(__file__).resolve().parent
    if (script_parent.parent / "data").exists():
        return script_parent.parent
    cwd = Path.cwd()
    if (cwd / "data").exists():
        return cwd
    if (cwd / "driver-safety-monitor" / "data").exists():
        return cwd / "driver-safety-monitor"
    return script_parent.parent


def count_existing_images(folder: Path) -> int:
    """Counts existing .jpg or .png image files in a folder."""
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def draw_hud(frame, label_counts, active_label=None):
    """Renders on-screen card with saved count per label and keymap guide."""
    h, w = frame.shape[:2]

    # Semi-transparent overlay card on top-left
    card_w = 340
    card_h = 240
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (15 + card_w, 15 + card_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (15, 15), (15 + card_w, 15 + card_h), (80, 80, 80), 1)

    # Title
    cv2.putText(
        frame,
        "DATASET RECORDER",
        (25, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )

    # Counts per label
    y_pos = 72
    for label, count in label_counts.items():
        key_char = LABEL_KEY_DISPLAY.get(label, "?")
        is_active = (active_label == label)

        # Highlight line if actively saving
        line_color = (0, 255, 0) if is_active else (220, 220, 220)
        prefix = "> " if is_active else "  "

        text = f"{prefix}[{key_char}] {label:<22} : {count:>4}"
        cv2.putText(
            frame,
            text,
            (25, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            line_color,
            1,
            cv2.LINE_AA,
        )
        y_pos += 26

    # Bottom helper message
    cv2.putText(
        frame,
        "Press 'q' to quit",
        (25, y_pos + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    # Top banner indicator when recording
    if active_label:
        banner = f"SAVING: {active_label.upper()}"
        cv2.rectangle(frame, (w - 320, 15), (w - 20, 55), (0, 0, 180), -1)
        cv2.circle(frame, (w - 295, 35), 8, (0, 0, 255), -1)
        cv2.putText(
            frame,
            banner,
            (w - 275, 41),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


def run_recorder(camera_index: int = 0):
    base_dir = get_base_dir()
    own_data_dir = base_dir / "data" / "own"

    # Create directories for each label
    label_dirs = {}
    label_counts = {}
    for label in KEY_LABEL_MAP.values():
        dir_path = own_data_dir / label
        dir_path.mkdir(parents=True, exist_ok=True)
        label_dirs[label] = dir_path
        label_counts[label] = count_existing_images(dir_path)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera at index {camera_index}. Check camera connection and permissions.")
        return

    window_name = "Driver Safety Monitor - Record Samples"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("=" * 65)
    print("              Custom Sample Recorder Started              ")
    print("=" * 65)
    print(f"Session ID : {session_id}")
    print(f"Save Path  : {own_data_dir}")
    print("\nControls:")
    print("  Hold 'o' : eyes_open")
    print("  Hold 'c' : eyes_closed")
    print("  Hold 'f' : facing_forward")
    print("  Hold 'a' : looking_away")
    print("  Hold 'd' : eyes_open_looking_down")
    print("  Press 'q': Quit and finish")
    print("=" * 65)

    session_saved = {label: 0 for label in KEY_LABEL_MAP.values()}
    frame_indices = {label: 0 for label in KEY_LABEL_MAP.values()}

    last_active_label = None
    last_active_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame from webcam.")
            break

        # Check key press (10ms wait time for smooth key capture)
        key = cv2.waitKey(10) & 0xFF

        active_label = None
        if key == ord("q"):
            print("\nExiting recorder...")
            break
        elif key in KEY_LABEL_MAP:
            active_label = KEY_LABEL_MAP[key]
            last_active_label = active_label
            last_active_time = time.time()

            # Save clean frame BEFORE drawing HUD overlay
            frame_indices[active_label] += 1
            idx = frame_indices[active_label]
            filename = f"{active_label}_{session_id}_{idx:05d}.jpg"
            save_path = label_dirs[active_label] / filename
            cv2.imwrite(str(save_path), frame)

            label_counts[active_label] += 1
            session_saved[active_label] += 1

        # Keep active indicator visible for a fraction of a second if key repeat has gaps
        display_active = active_label or (
            last_active_label if time.time() - last_active_time < 0.25 else None
        )

        # Draw HUD on display frame
        display_frame = frame.copy()
        draw_hud(display_frame, label_counts, active_label=display_active)

        cv2.imshow(window_name, display_frame)

    cap.release()
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)

    print("\n" + "=" * 65)
    print("               Recording Session Summary                  ")
    print("=" * 65)
    print(f"{'Label':<25} | {'Session Added':>15} | {'Total in Folder':>15}")
    print("-" * 65)
    for label in KEY_LABEL_MAP.values():
        print(f"{label:<25} | {session_saved[label]:>15} | {label_counts[label]:>15}")
    print("=" * 65)


def main():
    run_recorder(camera_index=0)


if __name__ == "__main__":
    main()
