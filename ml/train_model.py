"""
train_model.py — Train, evaluate, and save the RF drone detection model.

Pipeline:
  1. Load dataset (synthetic or real — same schema)
  2. StandardScaler fit on train set only
  3. GridSearchCV over Random Forest hyperparameters (F1 scoring)
  4. Precision-recall threshold tuning (enforce FPR < 10%)
  5. Evaluate on held-out test set
  6. Save model artifacts to models/

Outputs:
  models/drone_classifier.pkl   — trained sklearn Random Forest
  models/scaler_params.json     — scaler mean/std for firmware embedding
  models/threshold.json         — optimized decision threshold

Run:
  python ml/train_model.py
  python ml/train_model.py --data dataset/data/real_rf_features.csv
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Ensure ml/ package imports work from any working directory
sys.path.insert(0, str(Path(__file__).parent))

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "rssi_mean", "rssi_variance", "burst_count", "active_channel_count",
    "channel_hopping_rate", "peak_channel_index", "wifi_channel_ratio",
    "signal_duration",
]
LABEL_COL = "label"
MAX_FPR = 0.10  # reject threshold configs that exceed this

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int8)

    n_drone = y.sum()
    n_total = len(y)
    ratio = (n_total - n_drone) / max(n_drone, 1)
    if ratio > 5 or ratio < 0.2:
        print(f"  WARNING: Class imbalance ratio = {ratio:.1f}. Consider resampling.")

    print(f"  Loaded {n_total} samples — drone: {n_drone} ({n_drone/n_total*100:.1f}%), "
          f"no-drone: {n_total - n_drone} ({(n_total - n_drone)/n_total*100:.1f}%)")
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(X_train, y_train):
    print("\nRunning GridSearchCV (this may take a moment)...")

    param_grid = {
        "n_estimators": [16, 32, 64],
        "max_depth": [4, 6, 8, 10],
        "min_samples_leaf": [2, 4, 8],
    }

    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid = GridSearchCV(
        base_rf,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    print(f"  Best params  : {grid.best_params_}")
    print(f"  Best CV F1   : {grid.best_score_:.4f}")
    return grid.best_estimator_


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_threshold(model, X_val, y_val) -> float:
    """
    Find the lowest threshold t such that FPR <= MAX_FPR on the validation set.
    If no threshold achieves this, return the threshold that minimizes FPR.
    """
    probs = model.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    best_recall = 0.0
    best_fpr = 1.0

    for t in np.arange(0.1, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (y_val == 1)).sum()
        fp = ((preds == 1) & (y_val == 0)).sum()
        tn = ((preds == 0) & (y_val == 0)).sum()
        fn = ((preds == 0) & (y_val == 1)).sum()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if fpr <= MAX_FPR and recall > best_recall:
            best_recall = recall
            best_fpr = fpr
            best_threshold = t

    if best_recall == 0.0:
        # No threshold met FPR constraint — use minimum FPR threshold
        print(f"  WARNING: Could not achieve FPR <= {MAX_FPR}. "
              f"Using threshold with minimum FPR.")
        for t in np.arange(0.9, 0.1, -0.01):
            preds = (probs >= t).astype(int)
            tp = ((preds == 1) & (y_val == 1)).sum()
            fp = ((preds == 1) & (y_val == 0)).sum()
            tn = ((preds == 0) & (y_val == 0)).sum()
            fn = ((preds == 0) & (y_val == 1)).sum()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if fpr < best_fpr:
                best_fpr = fpr
                best_recall = recall
                best_threshold = t

    print(f"  Tuned threshold: {best_threshold:.2f} "
          f"(val recall={best_recall:.4f}, val FPR={best_fpr:.4f})")
    return float(best_threshold)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test, threshold: float):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    tp = ((preds == 1) & (y_test == 1)).sum()
    fp = ((preds == 1) & (y_test == 0)).sum()
    tn = ((preds == 0) & (y_test == 0)).sum()
    fn = ((preds == 0) & (y_test == 1)).sum()

    accuracy  = (tp + tn) / len(y_test)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    roc_auc   = roc_auc_score(y_test, probs)

    print("\n--- Test Set Evaluation ---")
    print(f"  Threshold : {threshold:.2f}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  FPR       : {fpr:.4f}  (target < {MAX_FPR})")
    print(f"  ROC-AUC   : {roc_auc:.4f}")

    cm = confusion_matrix(y_test, preds)
    print(f"\n  Confusion Matrix:")
    print(f"                  Pred 0   Pred 1")
    print(f"  Actual 0 (neg)  {cm[0,0]:6d}   {cm[0,1]:6d}   <- FP on right")
    print(f"  Actual 1 (pos)  {cm[1,0]:6d}   {cm[1,1]:6d}   <- FN on left")

    if fpr > MAX_FPR:
        print(f"\n  !!! FPR {fpr:.4f} exceeds target {MAX_FPR}. Consider collecting "
              "more diverse no-drone training data or adjusting threshold. !!!")

    print("\n  Feature Importances:")
    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
    for name, imp in ranked:
        bar = "#" * int(imp * 40)
        print(f"    {name:<24} {imp:.4f}  {bar}")

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "fpr": fpr, "roc_auc": roc_auc,
    }


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(model, scaler, threshold: float, metrics: dict):
    # 1. Trained model
    model_path = MODELS_DIR / "drone_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved model     : {model_path}")

    # 2. Scaler parameters (for firmware embedding)
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "std":  scaler.scale_.tolist(),
        "features": FEATURE_COLS,
    }
    scaler_path = MODELS_DIR / "scaler_params.json"
    with open(scaler_path, "w") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"Saved scaler    : {scaler_path}")

    # 3. Decision threshold
    threshold_path = MODELS_DIR / "threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({"threshold": threshold, "metrics": metrics}, f, indent=2)
    print(f"Saved threshold : {threshold_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train RF drone detection model")
    parser.add_argument(
        "--data",
        default=str(Path(__file__).parent.parent / "dataset" / "data" / "synthetic_rf_features.csv"),
        help="Path to feature CSV (default: dataset/data/synthetic_rf_features.csv)",
    )
    args = parser.parse_args()

    csv_path = Path(args.data)
    if not csv_path.exists():
        print(f"ERROR: Dataset not found at {csv_path}")
        print("Run: python dataset/generate_synthetic_dataset.py")
        sys.exit(1)

    print(f"Loading dataset: {csv_path}")
    X, y = load_dataset(csv_path)

    # Stratified splits: 70% train, 15% val, 15% test
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(splitter.split(X, y))
    X_temp, y_temp = X[temp_idx], y[temp_idx]

    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(splitter2.split(X_temp, y_temp))
    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X_temp[val_idx],  y_temp[val_idx]
    X_test,  y_test  = X_temp[test_idx], y_temp[test_idx]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # Train
    model = train(X_train_s, y_train)

    # Threshold tuning on validation set
    threshold = tune_threshold(model, X_val_s, y_val)

    # Final evaluation on test set
    metrics = evaluate(model, X_test_s, y_test, threshold)

    # Save everything
    save_artifacts(model, scaler, threshold, metrics)

    print("\nTraining complete. Next step: run ml/export_to_firmware.py")


if __name__ == "__main__":
    main()
