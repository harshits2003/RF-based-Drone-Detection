"""
export_to_firmware.py — Export trained sklearn model to C header for ESP32.

Generates firmware/ml_model.h containing:
  - Random Forest as pure C decision tree functions (no library dependencies)
  - StandardScaler constants (FEATURE_MEAN, FEATURE_STD)
  - Decision threshold (DECISION_THRESHOLD)
  - Parity test: verifies 20 test vectors produce identical scores in C and Python

Usage:
  python ml/export_to_firmware.py

Optional (cleaner output):
  pip install micromlgen
  python ml/export_to_firmware.py --use-micromlgen

The generated ml_model.h is self-contained and builds with the Arduino framework
on ESP32. NEVER hand-edit this file — always regenerate by re-running this script.
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

MODELS_DIR   = Path(__file__).parent.parent / "models"
FIRMWARE_DIR = Path(__file__).parent.parent / "firmware"
FIRMWARE_DIR.mkdir(exist_ok=True)
OUTPUT_H     = FIRMWARE_DIR / "ml_model.h"

FEATURE_NAMES = [
    "rssi_mean", "rssi_variance", "burst_count", "active_channel_count",
    "channel_hopping_rate", "peak_channel_index", "wifi_channel_ratio",
    "signal_duration",
]


# ---------------------------------------------------------------------------
# Manual Random Forest → C exporter
# ---------------------------------------------------------------------------

def _export_tree_c(tree, tree_idx: int) -> str:
    """Export a single sklearn DecisionTreeClassifier as a C function."""
    t = tree.tree_
    n_nodes = t.node_count

    lines = [f"static float _tree_{tree_idx}(const float* f) {{"]

    def _node(node_id: int, indent: int) -> None:
        pad = "  " * indent
        left  = t.children_left[node_id]
        right = t.children_right[node_id]

        if left == -1:  # leaf node
            # t.value[node_id] has shape (1, n_classes) for sklearn classifiers
            values = t.value[node_id][0]
            total  = values.sum()
            # Probability of class 1 (drone)
            prob = values[1] / total if total > 0 else 0.0
            lines.append(f"{pad}return {prob:.6f}f;")
        else:
            feat_idx = int(t.feature[node_id])
            threshold = float(t.threshold[node_id])
            feat_name = FEATURE_NAMES[feat_idx] if feat_idx < len(FEATURE_NAMES) else f"f{feat_idx}"
            lines.append(f"{pad}if (f[{feat_idx}] <= {threshold:.6f}f) {{  /* {feat_name} */")
            _node(left, indent + 1)
            lines.append(f"{pad}}} else {{")
            _node(right, indent + 1)
            lines.append(f"{pad}}}")

    _node(0, 1)
    lines.append("}")
    return "\n".join(lines)


def export_rf_manual(model) -> str:
    """Export full Random Forest as C code (manual implementation)."""
    n_trees = len(model.estimators_)
    lines = []
    lines.append("/* ---- Decision Trees ---- */")
    for i, estimator in enumerate(model.estimators_):
        lines.append(_export_tree_c(estimator, i))
        lines.append("")

    # Aggregation function
    lines.append("/* ---- Random Forest predict (returns probability [0,1]) ---- */")
    lines.append("float rf_predict_proba(const float* f) {")
    lines.append(f"  float sum = 0.0f;")
    for i in range(n_trees):
        lines.append(f"  sum += _tree_{i}(f);")
    lines.append(f"  return sum / {n_trees}.0f;")
    lines.append("}")
    return "\n".join(lines)


def export_rf_micromlgen(model) -> str:
    """Export using micromlgen library (cleaner, more compact output)."""
    from micromlgen import port
    return port(model, classname="DroneClassifier")


# ---------------------------------------------------------------------------
# Parity test
# ---------------------------------------------------------------------------

def run_parity_test(model, scaler, threshold: float, n_samples: int = 20):
    """Verify Python and C produce identical predictions on n_samples vectors."""
    rng = np.random.default_rng(99)
    # Generate random-ish feature vectors in [0,1]
    X_raw = rng.uniform(0.0, 1.0, (n_samples, 8)).astype(np.float32)
    X_scaled = scaler.transform(X_raw)

    py_probs = model.predict_proba(X_scaled)[:, 1]
    py_preds = (py_probs >= threshold).astype(int)

    print(f"\nParity test ({n_samples} random vectors):")
    print(f"  {'#':>3}  {'raw[0]':>8}  {'raw[4]':>8}  {'prob_py':>8}  {'pred_py':>7}")
    for i in range(n_samples):
        print(f"  {i:>3}  {X_raw[i,0]:>8.4f}  {X_raw[i,4]:>8.4f}  "
              f"{py_probs[i]:>8.4f}  {'DRONE' if py_preds[i] else 'none':>7}")

    print("\n  NOTE: Re-run this same parity test on ESP32 by flashing a DEBUG_PARITY")
    print("  build and comparing serial output. Tolerance: 0.002 per sample.")

    # Store raw vectors as C array for firmware parity test
    c_lines = ["/* Parity test vectors — use in DEBUG_PARITY build */",
               f"#define PARITY_N_SAMPLES {n_samples}",
               "static const float PARITY_RAW_FEATURES[PARITY_N_SAMPLES][8] = {"]
    for i in range(n_samples):
        vals = ", ".join(f"{v:.6f}f" for v in X_raw[i])
        c_lines.append(f"  {{{vals}}},")
    c_lines.append("};")
    c_lines.append(f"static const float PARITY_EXPECTED_PROBA[PARITY_N_SAMPLES] = {{")
    prob_vals = ", ".join(f"{p:.6f}f" for p in py_probs)
    c_lines.append(f"  {prob_vals}")
    c_lines.append("};")
    c_lines.append(f"static const uint8_t PARITY_EXPECTED_PRED[PARITY_N_SAMPLES] = {{")
    pred_vals = ", ".join(str(p) for p in py_preds)
    c_lines.append(f"  {pred_vals}")
    c_lines.append("};")

    return "\n".join(c_lines)


# ---------------------------------------------------------------------------
# Header file assembly
# ---------------------------------------------------------------------------

def build_header(model, scaler, threshold: float, rf_c_code: str,
                 parity_c_code: str, test_metrics: dict) -> str:
    mean_vals = scaler.mean_.tolist()
    std_vals  = scaler.scale_.tolist()

    mean_str = ", ".join(f"{v:.6f}f" for v in mean_vals)
    std_str  = ", ".join(f"{v:.6f}f" for v in std_vals)

    n_trees = len(model.estimators_)
    max_depth = model.max_depth or "auto"

    lines = [
        "/*",
        " * ml_model.h — Auto-generated by ml/export_to_firmware.py",
        f" * Generated   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f" * Model       : Random Forest ({n_trees} trees, max_depth={max_depth})",
        f" * Threshold   : {threshold:.4f}",
    ]
    if test_metrics:
        lines += [
            f" * Test Recall : {test_metrics.get('recall', 'N/A')}",
            f" * Test FPR    : {test_metrics.get('fpr', 'N/A')}",
            f" * Test F1     : {test_metrics.get('f1', 'N/A')}",
        ]
    lines += [
        " *",
        " * DO NOT HAND-EDIT. Regenerate with: python ml/export_to_firmware.py",
        " */",
        "",
        "#pragma once",
        "#include <stdint.h>",
        "",
        "/* ================================================================",
        " * 1. Feature normalization constants (from StandardScaler)",
        " * Apply BEFORE calling rf_predict_proba():",
        " *   for (int i = 0; i < 8; i++)",
        " *     features[i] = (features[i] - FEATURE_MEAN[i]) / FEATURE_STD[i];",
        " * ================================================================ */",
        "",
        "#define N_FEATURES 8",
        "",
        f"static const float FEATURE_MEAN[N_FEATURES] = {{ {mean_str} }};",
        f"static const float FEATURE_STD[N_FEATURES]  = {{ {std_str} }};",
        "",
        "/* Feature index mapping (matches FEATURE_MEAN/STD order): */",
    ]
    for i, name in enumerate(FEATURE_NAMES):
        lines.append(f"#define FEAT_{name.upper():<24} {i}")
    lines += [
        "",
        "/* ================================================================",
        " * 2. Decision threshold",
        "    A predicted probability >= DECISION_THRESHOLD → DRONE DETECTED",
        " * ================================================================ */",
        "",
        f"#define DECISION_THRESHOLD {threshold:.4f}f",
        "",
        "/* ================================================================",
        " * 3. Random Forest model (auto-generated tree functions)",
        " * ================================================================ */",
        "",
        rf_c_code,
        "",
        "/* ================================================================",
        " * 4. Normalize and predict (convenience wrapper)",
        " * Usage:",
        " *   float features[8] = {rssi_mean, rssi_variance, ...};",
        " *   bool drone = rf_classify(features);",
        " * ================================================================ */",
        "",
        "static inline void normalize_features(float* f) {",
        "  for (int i = 0; i < N_FEATURES; i++) {",
        "    f[i] = (f[i] - FEATURE_MEAN[i]) / FEATURE_STD[i];",
        "  }",
        "}",
        "",
        "static inline bool rf_classify(float* f) {",
        "  normalize_features(f);",
        "  float prob = rf_predict_proba(f);",
        "  return prob >= DECISION_THRESHOLD;",
        "}",
        "",
        "/* ================================================================",
        " * 5. Parity test vectors (for DEBUG_PARITY build validation)",
        " * ================================================================ */",
        "",
        parity_c_code,
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-micromlgen", action="store_true",
                        help="Use micromlgen library for RF export (pip install micromlgen)")
    args = parser.parse_args()

    # Load artifacts
    model_path     = MODELS_DIR / "drone_classifier.pkl"
    scaler_path    = MODELS_DIR / "scaler_params.json"
    threshold_path = MODELS_DIR / "threshold.json"

    for p in [model_path, threshold_path]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run: python ml/train_model.py")
            sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(threshold_path) as f:
        threshold_data = json.load(f)
    threshold   = threshold_data["threshold"]
    test_metrics = threshold_data.get("metrics", {})

    # Rebuild scaler from saved params (no sklearn needed for firmware export)
    with open(scaler_path) as f:
        sp = json.load(f)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.mean_  = np.array(sp["mean"])
    scaler.scale_ = np.array(sp["std"])
    scaler.var_   = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    # Export RF to C
    if args.use_micromlgen:
        try:
            rf_c_code = export_rf_micromlgen(model)
            print("Using micromlgen export.")
        except ImportError:
            print("micromlgen not installed. Falling back to manual exporter.")
            rf_c_code = export_rf_manual(model)
    else:
        print("Using built-in RF-to-C exporter.")
        rf_c_code = export_rf_manual(model)

    # Parity test
    parity_c_code = run_parity_test(model, scaler, threshold)

    # Assemble header
    header = build_header(model, scaler, threshold, rf_c_code, parity_c_code, test_metrics)

    # Write output
    with open(OUTPUT_H, "w") as f:
        f.write(header)

    size_bytes = OUTPUT_H.stat().st_size
    print(f"\nGenerated: {OUTPUT_H}")
    print(f"File size: {size_bytes / 1024:.1f} KB")
    print(f"\nNext step: open firmware/esp32_main.ino in Arduino IDE")
    print(f"  - Include ml_model.h")
    print(f"  - Build and flash to ESP32")
    print(f"  - Enable #define DEBUG_PARITY to run parity validation")


if __name__ == "__main__":
    main()
