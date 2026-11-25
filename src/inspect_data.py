import os
import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_base_dir() -> Path:
    """Finds the project root directory whether run from project root or src."""
    script_parent = Path(__file__).resolve().parent
    if (script_parent.parent / "data").exists():
        return script_parent.parent
    cwd = Path.cwd()
    if (cwd / "data").exists():
        return cwd
    if (cwd / "driver-safety-monitor" / "data").exists():
        return cwd / "driver-safety-monitor"
    return script_parent.parent


def get_image_files(folder_path: Path):
    """Returns a list of image paths within a directory (non-recursive)."""
    if not folder_path.is_dir():
        return []
    return [
        p for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]


def inspect_dataset(data_dir: Path, results_dir: Path, samples_per_class: int = 4):
    print("=" * 60)
    print("           Dataset Inspection: data/raw/dataset_new           ")
    print("=" * 60)
    print(f"Dataset path: {data_dir}")

    splits = ["train", "test"]
    split_dirs = {s: data_dir / s for s in splits}

    for s, path in split_dirs.items():
        if not path.exists():
            print(f"[ERROR] Split directory '{s}' not found at: {path}")
            return

    # Discover classes across train and test
    classes = sorted(list({
        d.name for s in splits for d in split_dirs[s].iterdir() if d.is_dir() and not d.name.startswith(".")
    }))

    if not classes:
        print("[ERROR] No class directories found in train/test splits.")
        return

    # Count images per class per split
    counts = {cls: {} for cls in classes}
    all_images_by_class = {cls: [] for cls in classes}

    for s in splits:
        for cls in classes:
            cls_dir = split_dirs[s] / cls
            images = get_image_files(cls_dir)
            counts[cls][s] = len(images)
            all_images_by_class[cls].extend(images)

    # Print summary table
    print("\n" + "-" * 60)
    print(f"{'Class':<20} | {'Train':>10} | {'Test':>10} | {'Total':>10}")
    print("-" * 60)

    total_train = sum(counts[c].get("train", 0) for c in classes)
    total_test = sum(counts[c].get("test", 0) for c in classes)

    for cls in classes:
        train_cnt = counts[cls].get("train", 0)
        test_cnt = counts[cls].get("test", 0)
        total_cnt = train_cnt + test_cnt
        print(f"{cls:<20} | {train_cnt:>10} | {test_cnt:>10} | {total_cnt:>10}")

    print("-" * 60)
    print(f"{'TOTAL':<20} | {total_train:>10} | {total_test:>10} | {total_train + total_test:>10}")
    print("-" * 60)

    # Create sample grid: num_classes rows x samples_per_class columns
    num_classes = len(classes)
    fig, axes = plt.subplots(
        nrows=num_classes,
        ncols=samples_per_class,
        figsize=(samples_per_class * 3.2, num_classes * 3.0),
    )

    # Ensure axes is 2D array even for edge cases (1 class)
    if num_classes == 1:
        axes = [axes]

    random.seed(42)  # For reproducible sample selection

    for row_idx, cls in enumerate(classes):
        # Prefer taking samples from 'train', fallback to all images for the class
        train_cls_dir = split_dirs["train"] / cls
        candidates = get_image_files(train_cls_dir)
        if len(candidates) < samples_per_class:
            candidates = all_images_by_class[cls]

        sampled_paths = random.sample(candidates, min(len(candidates), samples_per_class)) if candidates else []

        for col_idx in range(samples_per_class):
            ax = axes[row_idx][col_idx]
            ax.axis("off")

            if col_idx < len(sampled_paths):
                img_path = sampled_paths[col_idx]
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                    ax.set_title(f"{cls}\n({img_path.name})", fontsize=10, pad=5)
                else:
                    ax.text(0.5, 0.5, "Failed to load", ha="center", va="center", color="red")
            else:
                ax.text(0.5, 0.5, "No image", ha="center", va="center", color="gray")

    plt.suptitle("Dataset Samples per Class", fontsize=16, y=0.995, fontweight="bold")
    plt.tight_layout()

    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "samples.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n[SUCCESS] Grid of {samples_per_class} sample images per class saved to:")
    print(f"          -> {output_path}")


def main():
    base_dir = find_base_dir()
    data_dir = base_dir / "data" / "raw" / "dataset_new"
    results_dir = base_dir / "results"

    inspect_dataset(data_dir=data_dir, results_dir=results_dir, samples_per_class=4)


if __name__ == "__main__":
    main()
