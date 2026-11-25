"""
Dataset Builder for Driver Safety Monitoring.

Runs extract_features on two sources and saves data/features.csv:

Source 1 - dataset_new (yawn / no_yawn folders, train + test splits):
  Columns populated: EAR, MAR, yaw, pitch, source, session, yawn
  Labels unavailable: eyes_closed, looking_away (left blank)

Source 2 - data/own/ (user-recorded frames per label folder):
  Columns populated: EAR, MAR, yaw, pitch, source, session, eyes_closed, looking_away
  Label unavailable: yawn (left blank)
  own/ label → label columns mapping:
    eyes_open                → eyes_closed=0
    eyes_closed              → eyes_closed=1
    facing_forward           → looking_away=0
    looking_away             → looking_away=1
    eyes_open_looking_down   → eyes_closed=0, looking_away=1

Frames where no face is detected are skipped and counted.
"""

import sys
from pathlib import Path

import cv2
import pandas as pd

# Ensure src/ is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from features import extract_features

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Own data label → label column values
OWN_LABEL_MAP = {
    "eyes_open":               {"eyes_closed": 0},
    "eyes_closed":             {"eyes_closed": 1},
    "facing_forward":          {"looking_away": 0, "looking_down": 0},
    "looking_away":            {"looking_away": 1},
    "eyes_open_looking_down":  {"eyes_closed": 0, "looking_down": 1},
}


def get_base_dir() -> Path:
    if (SCRIPT_DIR.parent / "data").exists():
        return SCRIPT_DIR.parent
    cwd = Path.cwd()
    if (cwd / "data").exists():
        return cwd
    if (cwd / "driver-safety-monitor" / "data").exists():
        return cwd / "driver-safety-monitor"
    return SCRIPT_DIR.parent


def extract_session_from_filename(stem: str, label: str) -> str:
    """
    Extracts the session timestamp from a filename stem.
    Pattern: {label}_{YYYYMMDD}_{HHMMSS}_{index}
    Example: eyes_open_20261007_191300_00042 → session = 20261007_191300
    """
    # Strip the label prefix and the following underscore
    remainder = stem[len(label):].lstrip("_")
    # Remaining: YYYYMMDD_HHMMSS_index → take first two parts
    parts = remainder.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return remainder


def process_image(img_path: Path):
    """Loads an image and runs extract_features. Returns features tuple or None."""
    frame = cv2.imread(str(img_path))
    if frame is None:
        return None
    return extract_features(frame)


def process_dataset_new(data_dir: Path) -> tuple[list[dict], int]:
    """
    Processes yawn and no_yawn folders from dataset_new (train + test splits).
    Returns (rows, skip_count).
    """
    rows = []
    skipped = 0
    splits = ["train", "test"]
    label_map = {"yawn": 1, "no_yawn": 0}

    for split in splits:
        for folder_name, yawn_val in label_map.items():
            folder = data_dir / split / folder_name
            if not folder.exists():
                print(f"  [WARN] Not found: {folder}")
                continue
            images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
            print(f"  Processing dataset_new/{split}/{folder_name}: {len(images)} images...", flush=True)
            for img_path in images:
                feats = process_image(img_path)
                if feats is None:
                    skipped += 1
                    continue
                ear, mar, yaw, pitch = feats
                rows.append({
                    "EAR": ear,
                    "MAR": mar,
                    "yaw": yaw,
                    "pitch": pitch,
                    "source": "dataset_new",
                    "session": f"{split}/{folder_name}",
                    "yawn": yawn_val,
                    "eyes_closed": "",
                    "looking_away": "",
                    "looking_down": "",
                })

    return rows, skipped


def process_own_data(own_dir: Path) -> tuple[list[dict], int]:
    """
    Processes all label subdirectories under data/own/.
    Returns (rows, skip_count).
    """
    rows = []
    skipped = 0

    for label, label_cols in OWN_LABEL_MAP.items():
        folder = own_dir / label
        if not folder.exists():
            print(f"  [SKIP] data/own/{label} not found (no recordings yet)")
            continue

        images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
        print(f"  Processing own/{label}: {len(images)} images...", flush=True)

        for img_path in images:
            feats = process_image(img_path)
            if feats is None:
                skipped += 1
                continue
            ear, mar, yaw, pitch = feats

            session = extract_session_from_filename(img_path.stem, label)

            row = {
                "EAR": ear,
                "MAR": mar,
                "yaw": yaw,
                "pitch": pitch,
                "source": "own",
                "session": session,
                "yawn": "",
                "eyes_closed": "",
                "looking_away": "",
                "looking_down": "",
            }
            # Fill in available label columns
            for col, val in label_cols.items():
                row[col] = val

            rows.append(row)

    return rows, skipped


def main():
    base_dir = get_base_dir()
    dataset_new_dir = base_dir / "data" / "raw" / "dataset_new"
    own_dir = base_dir / "data" / "own"
    output_csv = base_dir / "data" / "features.csv"

    print("=" * 65)
    print("            Dataset Feature Builder                       ")
    print("=" * 65)
    print(f"Base dir    : {base_dir}")
    print(f"dataset_new : {dataset_new_dir}")
    print(f"own data    : {own_dir}")
    print(f"Output CSV  : {output_csv}")
    print()

    all_rows = []
    total_skipped = 0

    # --- Source 1: dataset_new ---
    print("[1/2] Processing dataset_new (yawn / no_yawn)...")
    if dataset_new_dir.exists():
        rows1, skip1 = process_dataset_new(dataset_new_dir)
        all_rows.extend(rows1)
        total_skipped += skip1
        print(f"      -> {len(rows1)} rows extracted, {skip1} frames skipped (no face)")
    else:
        print(f"  [WARN] dataset_new not found at {dataset_new_dir}. Skipping.")
    print()

    # --- Source 2: data/own/ ---
    print("[2/2] Processing own recordings (data/own/)...")
    if own_dir.exists():
        rows2, skip2 = process_own_data(own_dir)
        all_rows.extend(rows2)
        total_skipped += skip2
        print(f"      -> {len(rows2)} rows extracted, {skip2} frames skipped (no face)")
    else:
        print(f"  [WARN] own/ data directory not found at {own_dir}. Skipping.")
    print()

    # --- Save CSV ---
    if not all_rows:
        print("[ERROR] No rows to save. Exiting.")
        return

    df = pd.DataFrame(all_rows, columns=["EAR", "MAR", "yaw", "pitch", "source", "session", "yawn", "eyes_closed", "looking_away", "looking_down"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    # --- Summary ---
    print("=" * 65)
    print("                     Summary                              ")
    print("=" * 65)
    print(f"Total rows saved   : {len(df)}")
    print(f"Total skipped      : {total_skipped} (no face detected)")
    print(f"Saved to           : {output_csv}")
    print()
    print("Rows per source:")
    for source, grp in df.groupby("source"):
        print(f"  {source:<15}: {len(grp)} rows")
    print()
    print("Label value counts:")
    for col in ["yawn", "eyes_closed", "looking_away", "looking_down"]:
        s = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
        if len(s) > 0:
            counts = s.value_counts().sort_index()
            print(f"  {col}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print("=" * 65)


if __name__ == "__main__":
    main()
