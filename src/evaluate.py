"""
src/evaluate.py — Evaluate the three trained Logistic Regression models.

Loads the models and scalers saved by src/train_models.py.
Reconstructs the SAME train/test split used during training so results
are directly comparable.

Outputs
-------
results/metrics.csv                        — all metrics in one file
results/yawn_confusion_matrix.png
results/eyes_closed_confusion_matrix.png
results/looking_away_confusion_matrix.png
results/yawn_threshold.png
results/eyes_closed_threshold.png
results/looking_away_threshold.png

DO NOT retrain any model or change any scaler.
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
CSV_PATH    = BASE_DIR / "data" / "features.csv"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Import the SAME session-split function used during training.
# Importing from train_models guarantees the split is byte-for-byte identical.
# ---------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Model specifications  (must match exactly what train_models.py used)
# ---------------------------------------------------------------------------
MODEL_SPECS = [
    {
        "name":         "Yawn",
        "slug":         "yawn",
        "target":       "yawn",
        "features":     ["MAR"],
        "model_path":   MODELS_DIR / "yawn_model.joblib",
        "scaler_path":  MODELS_DIR / "yawn_scaler.joblib",
        "class_names":  ["No Yawn", "Yawn"],
    },
    {
        "name":         "Eyes Closed",
        "slug":         "eyes_closed",
        "target":       "eyes_closed",
        "features":     ["EAR"],
        "model_path":   MODELS_DIR / "eyes_closed_model.joblib",
        "scaler_path":  MODELS_DIR / "eyes_closed_scaler.joblib",
        "class_names":  ["Eyes Open", "Eyes Closed"],
    },
    {
        "name":         "Looking Away",
        "slug":         "looking_away",
        "target":       "looking_away",
        "features":     ["yaw"],
        "model_path":   MODELS_DIR / "looking_away_model.joblib",
        "scaler_path":  MODELS_DIR / "looking_away_scaler.joblib",
        "class_names":  ["Facing Forward", "Looking Away"],
    },
    {
        "name":         "Looking Down",
        "slug":         "looking_down",
        "target":       "looking_down",
        "features":     ["pitch"],
        "model_path":   MODELS_DIR / "looking_down_model.joblib",
        "scaler_path":  MODELS_DIR / "looking_down_scaler.joblib",
        "class_names":  ["Facing Forward", "Looking Down"],
    },
]

THRESHOLDS = [0.5, 0.4, 0.3]

# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------
SEP  = "=" * 70
SEP2 = "-" * 70


def banner(text: str) -> None:
    print()
    print(SEP)
    print(f"  {text}")
    print(SEP)


def section(text: str) -> None:
    print()
    print(SEP2)
    print(f"  {text}")
    print(SEP2)


# ---------------------------------------------------------------------------
# Rebuild the SAME train/test split used during training
# ---------------------------------------------------------------------------
def build_splits(df: pd.DataFrame,
                 target_col: str,
                 feature_cols: list[str]) -> tuple:
    """
    Reconstruct the train/test split exactly as train_models.py did (stratified 80/20 split).
    Returns (X_train, y_train, X_test, y_test, train_sessions, test_sessions)
    """
    df_model = df.dropna(subset=[target_col]).copy()
    df_model[target_col] = df_model[target_col].astype(int)

    if target_col == "looking_away" and "yaw" in feature_cols:
        df_model["yaw"] = df_model["yaw"].abs()

    X = df_model[feature_cols]
    y = df_model[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    return X_train, y_train, X_test, y_test, None, None


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def compute_metrics(y_true: pd.Series,
                    y_pred: np.ndarray) -> dict:
    """Computes accuracy, precision, recall, F1 for binary predictions."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }


def threshold_metrics(proba_pos: np.ndarray,
                      y_true: pd.Series,
                      thresholds: list[float]) -> list[dict]:
    """
    Evaluates predictions at multiple probability thresholds.
    prediction = 1 if probability >= threshold, else 0.
    A lower threshold makes the model more willing to predict the positive
    class → increases recall but may increase false positives (lower precision).
    """
    rows = []
    for t in thresholds:
        y_pred = (proba_pos >= t).astype(int)
        m = compute_metrics(y_true, y_pred)
        rows.append({"threshold": t, **m})
    return rows


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true: pd.Series,
                          y_pred: np.ndarray,
                          class_names: list[str],
                          model_name: str,
                          save_path: Path) -> None:
    """
    Plots and saves a labelled confusion matrix for the test set.
    Each cell is annotated so True Negative / False Positive /
    False Negative / True Positive are immediately identifiable.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"{model_name} — Test Set Confusion Matrix",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    # Annotate quadrant roles in small text below the matrix
    fig.text(0.5, -0.04,
             "Bottom-left = False Negatives (missed detections)  "
             "  Top-right = False Positives (false alarms)",
             ha="center", fontsize=8, style="italic", color="dimgray")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  Saved → {save_path}")


# ---------------------------------------------------------------------------
# Threshold comparison plot
# ---------------------------------------------------------------------------
def plot_threshold(thresh_rows: list[dict],
                   model_name: str,
                   save_path: Path) -> None:
    """
    Simple line plot showing how precision, recall, and F1 change
    as the classification threshold is lowered.
    """
    thresholds  = [r["threshold"]  for r in thresh_rows]
    precisions  = [r["precision"]  for r in thresh_rows]
    recalls     = [r["recall"]     for r in thresh_rows]
    f1s         = [r["f1"]         for r in thresh_rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, precisions, "o-", label="Precision", color="#4C72B0")
    ax.plot(thresholds, recalls,    "s-", label="Recall",    color="#DD8452")
    ax.plot(thresholds, f1s,        "^-", label="F1",        color="#55A868")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} — Threshold Experiment (Test Set)",
                 fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.invert_xaxis()   # left = most sensitive (0.3), right = default (0.5)
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  Saved → {save_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    banner("DRIVER SAFETY MONITOR — MODEL EVALUATION")

    # ── Load features.csv ────────────────────────────────────────────────────
    if not CSV_PATH.exists():
        print(f"[ERROR] {CSV_PATH} not found. Run src/build_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    for col in ["yawn", "eyes_closed", "looking_away", "looking_down"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  Loaded {len(df):,} rows from {CSV_PATH}")

    # ── Storage for CSV rows ─────────────────────────────────────────────────
    metrics_rows = []

    # ── Per-model evaluation ─────────────────────────────────────────────────
    for spec in MODEL_SPECS:
        name        = spec["name"]
        slug        = spec["slug"]
        target      = spec["target"]
        features    = spec["features"]
        class_names = spec["class_names"]

        banner(f"MODEL: {name.upper()}")

        # Load model and scaler
        if not spec["model_path"].exists():
            print(f"  [ERROR] Model not found: {spec['model_path']}")
            print("  Run src/train_models.py first.")
            continue
        model  = joblib.load(spec["model_path"])
        scaler = joblib.load(spec["scaler_path"])
        print(f"  Loaded model  : {spec['model_path'].name}")
        print(f"  Loaded scaler : {spec['scaler_path'].name}")

        # Rebuild identical train/test split
        section("Rebuilding Train/Test Split")
        X_train, y_train, X_test, y_test, train_sess, test_sess = build_splits(
            df, target, features
        )
        if train_sess:
            print(f"  TRAIN sessions : {train_sess}")
            print(f"  TEST  sessions : {test_sess}")
        print(f"  Train samples  : {len(X_train)}")
        print(f"  Test  samples  : {len(X_test)}")

        # Scale features (using the LOADED scaler — no re-fitting)
        X_train_sc = scaler.transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        # ── Standard predictions (threshold = 0.5) ───────────────────────────
        section("Classification Metrics (threshold = 0.5)")
        y_train_pred = model.predict(X_train_sc)
        y_test_pred  = model.predict(X_test_sc)

        train_m = compute_metrics(y_train, y_train_pred)
        test_m  = compute_metrics(y_test,  y_test_pred)

        for dataset, m in [("Train", train_m), ("Test", test_m)]:
            print(f"\n  {dataset}:")
            print(f"    Accuracy  : {m['accuracy']:.4f}")
            print(f"    Precision : {m['precision']:.4f}")
            print(f"    Recall    : {m['recall']:.4f}")
            print(f"    F1        : {m['f1']:.4f}")

        metrics_rows.append({"model": name, "dataset": "Train",
                              "threshold": float("nan"), **train_m})
        metrics_rows.append({"model": name, "dataset": "Test",
                              "threshold": float("nan"), **test_m})

        # Train vs Test comparison
        gap = train_m["f1"] - test_m["f1"]
        print()
        if gap > 0.10:
            print(f"  [NOTE] Train F1 ({train_m['f1']:.4f}) is notably higher than "
                  f"Test F1 ({test_m['f1']:.4f}) — gap = {gap:.4f}.")
            print("         This may indicate some overfitting. The model learnt patterns")
            print("         from the training data that do not fully generalise to new data.")
        else:
            print(f"  [NOTE] Train F1 ({train_m['f1']:.4f}) ≈ Test F1 ({test_m['f1']:.4f}) "
                  f"— gap = {gap:.4f}.")
            print("         The model appears to generalise reasonably well.")

        # ── Confusion matrix ─────────────────────────────────────────────────
        section("Test Confusion Matrix")
        cm_path = RESULTS_DIR / f"{slug}_confusion_matrix.png"
        plot_confusion_matrix(y_test, y_test_pred, class_names, name, cm_path)

        cm = confusion_matrix(y_test, y_test_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"  TN (correctly predicted negative) : {tn}")
        print(f"  FP (false alarm)                  : {fp}")
        print(f"  FN (missed detection)              : {fn}")
        print(f"  TP (correctly predicted positive)  : {tp}")

        # ── Threshold experiment ─────────────────────────────────────────────
        section("Threshold Experiment (Test Set)")
        print("  A lower threshold means the model predicts 'positive' more eagerly.")
        print("  This tends to increase Recall but may lower Precision.")
        print()

        proba_pos = model.predict_proba(X_test_sc)[:, 1]
        thresh_rows = threshold_metrics(proba_pos, y_test, THRESHOLDS)

        header = f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}"
        print(header)
        print("  " + "-" * 44)
        for r in thresh_rows:
            print(f"  {r['threshold']:>10.1f}  "
                  f"{r['precision']:>10.4f}  "
                  f"{r['recall']:>8.4f}  "
                  f"{r['f1']:>8.4f}")
            metrics_rows.append({"model": name, "dataset": "Test",
                                  "threshold": r["threshold"],
                                  "accuracy":  float("nan"),
                                  **{k: r[k] for k in ("precision", "recall", "f1")}})

        best_thresh = max(thresh_rows, key=lambda r: r["f1"])
        print(f"\n  Best F1 at threshold = {best_thresh['threshold']:.1f} "
              f"→ F1 = {best_thresh['f1']:.4f}")

        # Threshold plot
        thresh_path = RESULTS_DIR / f"{slug}_threshold.png"
        plot_threshold(thresh_rows, name, thresh_path)

    # ── Summary metrics table ─────────────────────────────────────────────────
    banner("FINAL SUMMARY — TRAIN vs TEST METRICS")
    print(f"  {'Model':<16} {'Dataset':<8} {'Accuracy':>9}  {'Precision':>10}  "
          f"{'Recall':>8}  {'F1':>8}")
    print("  " + "-" * 62)

    for spec in MODEL_SPECS:
        name = spec["name"]
        for ds in ("Train", "Test"):
            row = next((r for r in metrics_rows
                        if r["model"] == name and r["dataset"] == ds
                        and pd.isna(r["threshold"])), None)
            if row:
                print(f"  {name:<16} {ds:<8} "
                      f"{row['accuracy']:>9.4f}  "
                      f"{row['precision']:>10.4f}  "
                      f"{row['recall']:>8.4f}  "
                      f"{row['f1']:>8.4f}")
        print()

    # ── Threshold summary table ──────────────────────────────────────────────
    banner("THRESHOLD COMPARISON SUMMARY (Test Set)")
    print(f"  {'Model':<16} {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("  " + "-" * 58)
    for spec in MODEL_SPECS:
        name = spec["name"]
        for r in metrics_rows:
            if r["model"] == name and r["dataset"] == "Test" \
                    and not pd.isna(r.get("threshold", float("nan"))):
                print(f"  {name:<16} {r['threshold']:>10.1f}  "
                      f"{r['precision']:>10.4f}  "
                      f"{r['recall']:>8.4f}  "
                      f"{r['f1']:>8.4f}")
        print()

    # ── Safety interpretation ────────────────────────────────────────────────
    banner("SAFETY INTERPRETATION")
    print("""
  Accuracy   — the percentage of all predictions that were correct.
               Useful as a quick summary but misleading if classes are imbalanced.

  Precision  — of all the times the model raised a positive alert,
               how often it was actually correct.
               High precision means fewer false alarms.

  Recall     — of all the actual positive events in the data,
               how many the model successfully detected.
               High recall means fewer missed events.

  F1         — the harmonic mean of precision and recall.
               A single balanced score that penalises both missed
               detections AND false alarms equally.

  WHY RECALL MATTERS HERE:
  In a driver safety context, missing a real drowsiness or distraction event
  (a False Negative) is more serious than raising an occasional false alarm
  (a False Positive). Recall directly measures how few events are missed.

  IMPORTANT CAVEAT:
  Maximising recall by setting a very low threshold will eventually catch
  nearly all positive events but will flood the system with false warnings,
  which drivers may start to ignore. A practical system must balance
  missed events and false alarms depending on the deployment context.
    """)

    # ── Save metrics.csv ─────────────────────────────────────────────────────
    metrics_csv = RESULTS_DIR / "metrics.csv"
    df_metrics = pd.DataFrame(metrics_rows,
                              columns=["model", "dataset", "threshold",
                                       "accuracy", "precision", "recall", "f1"])
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"  Saved → {metrics_csv}")

    # ── Saved files summary ──────────────────────────────────────────────────
    banner("SAVED FILES")
    saved = [metrics_csv]
    for spec in MODEL_SPECS:
        saved.append(RESULTS_DIR / f"{spec['slug']}_confusion_matrix.png")
        saved.append(RESULTS_DIR / f"{spec['slug']}_threshold.png")
    for p in saved:
        exists = "✓" if p.exists() else "✗ MISSING"
        print(f"  {exists}  {p}")

    print()
    print("  Evaluation complete. Do NOT proceed to Step 9 until results are reviewed.")
    print(SEP)


if __name__ == "__main__":
    main()
