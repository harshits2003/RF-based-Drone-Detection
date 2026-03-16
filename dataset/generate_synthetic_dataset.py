"""
generate_synthetic_dataset.py — Create synthetic RF feature dataset.

Generates 5000 labeled samples across 5 RF environment classes:
  Class 0: Background noise only    (label=0)
  Class 1: WiFi only                (label=0)
  Class 2: Bluetooth only           (label=0)
  Class 3: Drone FHSS               (label=1)
  Class 4: Drone + WiFi mixed       (label=1)

Statistical distributions are designed to reflect real nRF24L01+ scanner
observations. Wide variance is intentional to improve generalization when
the model is later retrained with real RF captures.

Output CSV schema (matches collect_real_rf_data.py exactly):
  timestamp, rssi_mean, rssi_variance, burst_count, active_channel_count,
  channel_hopping_rate, peak_channel_index, wifi_channel_ratio,
  signal_duration, label

Run:
  python dataset/generate_synthetic_dataset.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Reproducible output
RNG = np.random.default_rng(2024)
SAMPLES_PER_CLASS = 1000
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "synthetic_rf_features.csv"


# ---------------------------------------------------------------------------
# Helper samplers
# ---------------------------------------------------------------------------

def _normal_clipped(mean, std, low, high, n, rng):
    """Sample from normal distribution, clipped to [low, high]."""
    s = rng.normal(mean, std, n)
    return np.clip(s, low, high)


def _beta_scaled(a, b, scale_min, scale_max, n, rng):
    """Sample from Beta(a,b), linearly scaled to [scale_min, scale_max]."""
    s = rng.beta(a, b, n)
    return scale_min + s * (scale_max - scale_min)


def _wifi_peak_index(n, rng):
    """
    Sample peak_channel_index for WiFi: mixture of 3 Gaussians centered at
    known WiFi channel indices normalized to [0,1]:
      ch1  -> idx 12  -> 12/124 = 0.097
      ch6  -> idx 37  -> 37/124 = 0.298
      ch11 -> idx 62  -> 62/124 = 0.500
    """
    choice = rng.integers(0, 3, n)
    centers = [0.097, 0.298, 0.500]
    out = np.empty(n)
    for i, c in enumerate(centers):
        mask = (choice == i)
        out[mask] = np.clip(rng.normal(c, 0.04, mask.sum()), 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Per-class samplers
# ---------------------------------------------------------------------------

def _class_background(n, rng):
    """Class 0: background noise — sparse, uncorrelated activity."""
    return {
        "rssi_mean":            _beta_scaled(1.5, 12, 0.0, 0.08, n, rng),
        "rssi_variance":        _beta_scaled(1.2, 10, 0.0, 0.06, n, rng),
        "burst_count":          _beta_scaled(1.2, 10, 0.0, 0.10, n, rng),
        "active_channel_count": _beta_scaled(1.5, 10, 0.0, 0.12, n, rng),
        "channel_hopping_rate": _beta_scaled(1.2, 15, 0.0, 0.04, n, rng),
        "peak_channel_index":   rng.uniform(0.0, 1.0, n),
        "wifi_channel_ratio":   _beta_scaled(1.2, 6,  0.0, 0.30, n, rng),
        "signal_duration":      _beta_scaled(1.2, 8,  0.0, 0.20, n, rng),
    }


def _class_wifi(n, rng):
    """Class 1: WiFi only — bursty, concentrated at known channel bands."""
    return {
        "rssi_mean":            _normal_clipped(0.12, 0.04, 0.04, 0.30, n, rng),
        "rssi_variance":        _normal_clipped(0.08, 0.025, 0.02, 0.18, n, rng),
        "burst_count":          _normal_clipped(0.36, 0.10, 0.08, 0.65, n, rng),  # many bursts
        "active_channel_count": _normal_clipped(0.12, 0.03, 0.06, 0.25, n, rng),
        "channel_hopping_rate": _normal_clipped(0.015, 0.006, 0.003, 0.04, n, rng),
        "peak_channel_index":   _wifi_peak_index(n, rng),
        "wifi_channel_ratio":   _normal_clipped(0.72, 0.10, 0.45, 0.95, n, rng),
        "signal_duration":      _normal_clipped(0.65, 0.15, 0.30, 0.95, n, rng),
    }


def _class_bluetooth(n, rng):
    """Class 2: Bluetooth — AFH hops across most of band, near-continuous."""
    return {
        "rssi_mean":            _normal_clipped(0.09, 0.03,  0.03, 0.20, n, rng),
        "rssi_variance":        _normal_clipped(0.04, 0.015, 0.01, 0.10, n, rng),
        "burst_count":          _normal_clipped(0.15, 0.05,  0.04, 0.35, n, rng),
        "active_channel_count": _normal_clipped(0.42, 0.08,  0.20, 0.65, n, rng),
        "channel_hopping_rate": _normal_clipped(0.08, 0.02,  0.04, 0.15, n, rng),
        "peak_channel_index":   rng.uniform(0.1, 0.9, n),
        "wifi_channel_ratio":   _normal_clipped(0.08, 0.04,  0.00, 0.18, n, rng),
        "signal_duration":      _normal_clipped(0.85, 0.08,  0.60, 0.99, n, rng),
    }


def _class_drone_fhss(n, rng):
    """
    Class 3: Drone FHSS — continuous hopping, sustained energy, not WiFi-aligned.
    channel_hopping_rate is the PRIMARY discriminating feature.
    """
    return {
        "rssi_mean":            _normal_clipped(0.18, 0.05,  0.08, 0.40, n, rng),
        "rssi_variance":        _normal_clipped(0.045, 0.015, 0.015, 0.09, n, rng),
        "burst_count":          _normal_clipped(0.10, 0.035,  0.02, 0.24, n, rng),
        "active_channel_count": _normal_clipped(0.35, 0.10,  0.12, 0.65, n, rng),
        "channel_hopping_rate": _normal_clipped(0.22, 0.06,  0.08, 0.40, n, rng),  # KEY
        "peak_channel_index":   _normal_clipped(0.45, 0.25,  0.10, 0.90, n, rng),
        "wifi_channel_ratio":   _normal_clipped(0.10, 0.05,  0.00, 0.22, n, rng),
        "signal_duration":      _normal_clipped(0.93, 0.05,  0.75, 1.00, n, rng),
    }


def _class_drone_wifi_mixed(n, rng):
    """
    Class 4: Drone + WiFi coexistence — blend of drone and WiFi features.
    Models the realistic scenario of a drone flying in a WiFi-dense environment.
    """
    drone = _class_drone_fhss(n, rng)
    wifi  = _class_wifi(n, rng)
    w_d = 0.55  # drone weight
    w_w = 0.45  # WiFi weight
    noise_std = 0.01

    mixed = {}
    for key in drone:
        blended = w_d * drone[key] + w_w * wifi[key]
        blended += rng.normal(0, noise_std, n)
        # Re-clip to [0, 1] after blending
        mixed[key] = np.clip(blended, 0.0, 1.0)
    return mixed


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

CLASS_GENERATORS = [
    (_class_background,      0, "background_noise"),
    (_class_wifi,            0, "wifi_only"),
    (_class_bluetooth,       0, "bluetooth_only"),
    (_class_drone_fhss,      1, "drone_fhss"),
    (_class_drone_wifi_mixed, 1, "drone_wifi_mixed"),
]


def generate_dataset(n_per_class: int = SAMPLES_PER_CLASS, seed: int = 2024) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for generator, label, class_name in CLASS_GENERATORS:
        features = generator(n_per_class, rng)
        for i in range(n_per_class):
            row = {k: float(v[i]) for k, v in features.items()}
            row["label"] = label
            row["class_name"] = class_name  # dropped before ML, kept for analysis
            rows.append(row)

    df = pd.DataFrame(rows)

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Add timestamp column (monotonically increasing fake timestamps)
    df.insert(0, "timestamp", pd.RangeIndex(len(df)) * 1.0)

    return df


def validate_dataset(df: pd.DataFrame):
    """Sanity-check the generated dataset to catch distribution issues."""
    print("\nDataset validation:")
    print(f"  Total samples : {len(df)}")
    print(f"  Drone (label=1): {(df['label'] == 1).sum()} ({(df['label'] == 1).mean()*100:.1f}%)")
    print(f"  No-drone (label=0): {(df['label'] == 0).sum()} ({(df['label'] == 0).mean()*100:.1f}%)")

    # Key discriminator: drone channel_hopping_rate should be significantly higher
    drone_hr = df[df['label'] == 1]['channel_hopping_rate'].mean()
    nodrone_hr = df[df['label'] == 0]['channel_hopping_rate'].mean()
    print(f"\n  channel_hopping_rate — drone mean: {drone_hr:.4f}, no-drone mean: {nodrone_hr:.4f}")
    assert drone_hr > nodrone_hr * 1.5, (
        f"WARN: Drone hop rate ({drone_hr:.4f}) not sufficiently higher than no-drone ({nodrone_hr:.4f})"
    )

    # WiFi ratio: no-drone should dominate
    drone_wr = df[df['label'] == 1]['wifi_channel_ratio'].mean()
    nodrone_wr = df[df['label'] == 0]['wifi_channel_ratio'].mean()
    print(f"  wifi_channel_ratio  — drone mean: {drone_wr:.4f}, no-drone mean: {nodrone_wr:.4f}")

    # Feature range checks
    feature_cols = [
        "rssi_mean", "rssi_variance", "burst_count", "active_channel_count",
        "channel_hopping_rate", "peak_channel_index", "wifi_channel_ratio",
        "signal_duration",
    ]
    out_of_range = False
    for col in feature_cols:
        mn, mx = df[col].min(), df[col].max()
        if mn < 0 or mx > 1:
            print(f"  WARN: {col} out of [0,1] range: min={mn:.4f}, max={mx:.4f}")
            out_of_range = True
    if not out_of_range:
        print("  All features within [0, 1] range.")

    print("\nValidation OK.\n")


def print_class_stats(df: pd.DataFrame):
    feature_cols = [
        "rssi_mean", "rssi_variance", "burst_count", "active_channel_count",
        "channel_hopping_rate", "peak_channel_index", "wifi_channel_ratio",
        "signal_duration",
    ]
    print("\nPer-class feature means:")
    print(f"{'Class':<24}", end="")
    for c in feature_cols:
        short = c[:10]
        print(f"  {short:>12}", end="")
    print()
    for _, group in df.groupby("class_name"):
        class_name = group["class_name"].iloc[0]
        label = group["label"].iloc[0]
        tag = f"{class_name} (L={label})"
        print(f"{tag:<24}", end="")
        for c in feature_cols:
            print(f"  {group[c].mean():>12.4f}", end="")
        print()


if __name__ == "__main__":
    print("Generating synthetic RF dataset...")
    df = generate_dataset()

    validate_dataset(df)
    print_class_stats(df)

    # Save — drop class_name before writing final CSV (matches schema)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out = df.drop(columns=["class_name"])
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df_out)} samples to: {OUTPUT_FILE}")
    print(f"Columns: {list(df_out.columns)}")
