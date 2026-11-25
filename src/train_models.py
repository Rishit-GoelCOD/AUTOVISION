"""
src/train_models.py — Simple & Clean Driver Safety Model Training.

Trains 4 fundamental Logistic Regression models for driver safety monitoring:
  1. Yawn Detection      — Feature: MAR (Mouth Aspect Ratio)
  2. Eye Closure         — Feature: EAR (Eye Aspect Ratio)
  3. Looking Away        — Feature: |yaw| (Horizontal nose deviation %)
  4. Looking Down        — Feature: pitch (Vertical nose position %)

Each model follows the standard University ML curriculum pipeline:
  Load Data → Train/Test Split (80/20) → Standard Scaling → Logistic Regression → Evaluation → Save Model
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths Setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "features.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Generic, Simple Model Trainer
# ---------------------------------------------------------------------------
def train_basic_model(
    model_name: str,
    feature_col: str,
    target_col: str,
    df: pd.DataFrame,
    use_abs_feature: bool = False,
    class_weight: str | None = "balanced",
) -> dict:
    """
    Trains a basic Logistic Regression classifier for a specific driver state.

    Steps:
      1. Filter rows containing valid labels for target_col.
      2. Split data into 80% Training and 20% Testing (stratified).
      3. Scale features using StandardScaler (fit on train, transform on test).
      4. Train LogisticRegression with balanced class weights.
      5. Evaluate on test set (Accuracy, Precision, Recall, F1).
      6. Save trained model and scaler to disk.
    """
    print("\n" + "=" * 60)
    print(f" TRAINING MODEL: {model_name.upper()} (Feature: {feature_col})")
    print("=" * 60)

    # 1. Prepare clean data subset
    subset = df.dropna(subset=[target_col]).copy()
    subset[target_col] = subset[target_col].astype(int)

    # If looking away, head turn left or right both represent deviation from center
    if use_abs_feature:
        X = subset[[feature_col]].abs()
    else:
        X = subset[[feature_col]]

    y = subset[target_col]

    print(f"Total labeled samples: {len(subset)}")
    print(f"Class distribution: {dict(y.value_counts())} (0=Normal, 1=Triggered)")

    # 2. Standard 80/20 Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")

    # 3. Feature Scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Basic Logistic Regression Model
    model = LogisticRegression(class_weight=class_weight, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate on Test Set
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Test Set Results ---")
    print(f"Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Triggered (1)"]))
    print(f"Confusion Matrix:\n{cm}")
    print(f"  TN={cm[0, 0]} (Normal correct), FP={cm[0, 1]} (False alarm)")
    print(f"  FN={cm[1, 0]} (Missed alert),   TP={cm[1, 1]} (Alert correct)")

    # 6. Save Model and Scaler
    model_file = MODELS_DIR / f"{model_name}_model.joblib"
    scaler_file = MODELS_DIR / f"{model_name}_scaler.joblib"
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    print(f"\nSaved model  -> {model_file.name}")
    print(f"Saved scaler -> {scaler_file.name}")

    return {"name": model_name, "accuracy": acc, "train_n": len(X_train), "test_n": len(X_test)}


# ---------------------------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------------------------
def main():
    print("============================================================")
    print("     DRIVER SAFETY MONITOR — BASIC MODEL TRAINING          ")
    print("============================================================")

    if not CSV_PATH.exists():
        print(f"[ERROR] Features file not found at {CSV_PATH}. Run build_dataset.py first.")
        return

    # Load dataset
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded dataset: {len(df):,} total records from {CSV_PATH.name}")

    # Train each of the 4 independent safety models
    results = []

    # 1. Yawn Detection (MAR)
    results.append(
        train_basic_model(
            model_name="yawn",
            feature_col="MAR",
            target_col="yawn",
            df=df,
        )
    )

    # 2. Eye Closure Detection (EAR)
    results.append(
        train_basic_model(
            model_name="eyes_closed",
            feature_col="EAR",
            target_col="eyes_closed",
            df=df,
        )
    )

    # 3. Looking Away Detection (Yaw / Horizontal Deviation)
    # class_weight=None gives wider natural bandwidth (~12-14% deviation) before alerting
    results.append(
        train_basic_model(
            model_name="looking_away",
            feature_col="yaw",
            target_col="looking_away",
            df=df,
            use_abs_feature=True,
            class_weight=None,
        )
    )

    # 4. Looking Down Detection (Pitch / Vertical Deviation)
    results.append(
        train_basic_model(
            model_name="looking_down",
            feature_col="pitch",
            target_col="looking_down",
            df=df,
        )
    )

    # Summary Table
    print("\n" + "=" * 60)
    print("                   TRAINING SUMMARY TABLE                   ")
    print("=" * 60)
    print(f"{'Model Name':<16} {'Feature':<12} {'Train N':>8} {'Test N':>8} {'Accuracy':>10}")
    print("-" * 60)
    features_list = ["MAR", "EAR", "|yaw|", "pitch"]
    for r, feat in zip(results, features_list):
        print(f"{r['name']:<16} {feat:<12} {r['train_n']:>8} {r['test_n']:>8} {r['accuracy']*100:>9.2f}%")
    print("=" * 60)
    print("All 4 basic models trained and ready for real-time monitoring!\n")


if __name__ == "__main__":
    main()
