"""
Verification script for facial feature extraction across yawn and no_yawn classes.

Randomly samples 30 images from:
- data/raw/dataset_new/train/no_yawn
- data/raw/dataset_new/train/yawn

Extracts EAR, MAR, yaw, and pitch using src/features.py, evaluates detection rate,
and prints comparative statistics (mean and standard deviation).
"""

import os
import random
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure src/ is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from features import extract_features

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_base_dir() -> Path:
    """Finds the driver-safety-monitor root directory."""
    if (SCRIPT_DIR.parent / "data").exists():
        return SCRIPT_DIR.parent
    cwd = Path.cwd()
    if (cwd / "data").exists():
        return cwd
    if (cwd / "driver-safety-monitor" / "data").exists():
        return cwd / "driver-safety-monitor"
    return SCRIPT_DIR.parent


def evaluate_class(class_name: str, folder_path: Path, sample_count: int = 30, seed: int = 42):
    """
    Evaluates feature extraction on random samples of a given class.
    """
    image_paths = [
        p for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    if not image_paths:
        print(f"[ERROR] No images found in {folder_path}")
        return None

    random.seed(seed)
    selected_paths = random.sample(image_paths, min(len(image_paths), sample_count))

    no_face_count = 0
    ears, mars, yaws, pitches = [], [], [], []

    for path in selected_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            no_face_count += 1
            continue

        feats = extract_features(frame)
        if feats is None:
            no_face_count += 1
        else:
            ear, mar, yaw, pitch = feats
            ears.append(ear)
            mars.append(mar)
            yaws.append(yaw)
            pitches.append(pitch)

    detected_count = len(selected_paths) - no_face_count

    stats = {
        "class": class_name,
        "total_sampled": len(selected_paths),
        "detected": detected_count,
        "no_face": no_face_count,
        "ear_mean": np.mean(ears) if ears else float("nan"),
        "ear_std": np.std(ears) if ears else float("nan"),
        "mar_mean": np.mean(mars) if mars else float("nan"),
        "mar_std": np.std(mars) if mars else float("nan"),
        "yaw_mean": np.mean(yaws) if yaws else float("nan"),
        "yaw_std": np.std(yaws) if yaws else float("nan"),
        "pitch_mean": np.mean(pitches) if pitches else float("nan"),
        "pitch_std": np.std(pitches) if pitches else float("nan"),
    }

    return stats


def main():
    base_dir = get_base_dir()
    data_dir = base_dir / "data" / "raw" / "dataset_new" / "train"

    yawn_dir = data_dir / "yawn"
    no_yawn_dir = data_dir / "no_yawn"

    print("=" * 65)
    print("        Feature Check: Yawn vs No-Yawn Comparison         ")
    print("=" * 65)
    print(f"Dataset root: {data_dir}\n")

    if not yawn_dir.exists() or not no_yawn_dir.exists():
        print(f"[ERROR] Required dataset folders not found:")
        print(f"  yawn    : {yawn_dir} (exists: {yawn_dir.exists()})")
        print(f"  no_yawn : {no_yawn_dir} (exists: {no_yawn_dir.exists()})")
        return

    sample_size = 30
    print(f"Extracting features from {sample_size} random images per class...\n")

    results = {}
    for cls_name, cls_dir in [("no_yawn", no_yawn_dir), ("yawn", yawn_dir)]:
        print(f"Processing '{cls_name}'...", end=" ", flush=True)
        stats = evaluate_class(cls_name, cls_dir, sample_count=sample_size, seed=42)
        results[cls_name] = stats
        print("done.")

    print("\n" + "=" * 65)
    print(f"{'Metric':<18} | {'no_yawn':>20} | {'yawn':>20}")
    print("=" * 65)

    ny = results["no_yawn"]
    y = results["yawn"]

    print(f"{'Total Sampled':<18} | {ny['total_sampled']:>20} | {y['total_sampled']:>20}")
    print(f"{'Faces Detected':<18} | {ny['detected']:>20} | {y['detected']:>20}")
    print(f"{'No Face Detected':<18} | {ny['no_face']:>20} | {y['no_face']:>20}")
    print("-" * 65)
    print(f"{'Mean EAR':<18} | {ny['ear_mean']:>14.4f} +/- {ny['ear_std']:<4.2f} | {y['ear_mean']:>14.4f} +/- {y['ear_std']:<4.2f}")
    print(f"{'Mean MAR':<18} | {ny['mar_mean']:>14.4f} +/- {ny['mar_std']:<4.2f} | {y['mar_mean']:>14.4f} +/- {y['mar_std']:<4.2f}")
    print(f"{'Mean Yaw (deg)':<18} | {ny['yaw_mean']:>14.2f} +/- {ny['yaw_std']:<4.2f} | {y['yaw_mean']:>14.2f} +/- {y['yaw_std']:<4.2f}")
    print(f"{'Mean Pitch (deg)':<18} | {ny['pitch_mean']:>14.2f} +/- {ny['pitch_std']:<4.2f} | {y['pitch_mean']:>14.2f} +/- {y['pitch_std']:<4.2f}")
    print("=" * 65)

    mar_ny = ny["mar_mean"]
    mar_y = y["mar_mean"]

    print("\n--- MAR Analysis ---")
    print(f"no_yawn Mean MAR : {mar_ny:.4f}")
    print(f"yawn Mean MAR    : {mar_y:.4f}")

    if mar_y > mar_ny:
        ratio = mar_y / mar_ny if mar_ny > 0 else float("inf")
        diff = mar_y - mar_ny
        print(f"[VERIFIED] Yawn MAR is {ratio:.2f}x higher (+{diff:.4f}) than no_yawn MAR.")
        print("           Mouth Aspect Ratio clearly distinguishes yawning from normal state!")
    else:
        print("[WARNING] Yawn MAR was not higher than no_yawn MAR. Review detection or samples.")


if __name__ == "__main__":
    main()
