"""
feature_extraction.py — Single source of truth for all 8 RF signal features.

Accepts a 2D binary sweep matrix (n_sweeps x 125 channels) and returns
a dict of 8 normalized features. This exact logic is ported to
firmware/feature_extraction.cpp — both must produce identical outputs.

Channel layout (nRF24L01+ scanner mode):
  Channel index 0  → 2.400 GHz
  Channel index 1  → 2.401 GHz
  ...
  Channel index 124 → 2.524 GHz

WiFi 2.4 GHz channel centers (802.11b/g/n):
  Ch 1  → 2.412 GHz → index 12
  Ch 6  → 2.437 GHz → index 37
  Ch 11 → 2.462 GHz → index 62

WiFi ±11 MHz bandwidth → ±11 index units per channel center.
  Ch 1  band: indices  1–23
  Ch 6  band: indices 26–48
  Ch 11 band: indices 51–73
"""

import numpy as np


# --- WiFi band index ranges (inclusive) ---
WIFI_BAND_RANGES = [
    (1,  23),   # WiFi channel 1
    (26, 48),   # WiFi channel 6
    (51, 73),   # WiFi channel 11
]
NUM_CHANNELS = 125
BURST_THRESHOLD = 2  # active channels per sweep to define a "burst"


def extract_features(sweep_matrix: np.ndarray) -> dict:
    """
    Compute 8 RF features from a binary sweep matrix.

    Args:
        sweep_matrix: np.ndarray of shape (n_sweeps, 125), dtype uint8 or bool.
                      Each element is 1 if RPD=1 on that channel/sweep, else 0.

    Returns:
        dict with keys: rssi_mean, rssi_variance, burst_count,
                        active_channel_count, channel_hopping_rate,
                        peak_channel_index, wifi_channel_ratio, signal_duration
    """
    sweep_matrix = np.array(sweep_matrix, dtype=np.float32)
    n_sweeps, n_ch = sweep_matrix.shape
    assert n_ch == NUM_CHANNELS, f"Expected 125 channels, got {n_ch}"
    assert n_sweeps >= 2, "Need at least 2 sweeps"

    # Per-sweep active channel counts
    per_sweep_counts = sweep_matrix.sum(axis=1)  # shape: (n_sweeps,)

    # 1. rssi_mean: normalized mean active channels per sweep
    rssi_mean = float(per_sweep_counts.mean() / NUM_CHANNELS)

    # 2. rssi_variance: normalized std of per-sweep active counts
    rssi_variance = float(per_sweep_counts.std() / NUM_CHANNELS)

    # 3. burst_count: number of threshold crossings (inactive→active transitions)
    #    A crossing is counted when consecutive sweeps cross BURST_THRESHOLD
    above = (per_sweep_counts > BURST_THRESHOLD).astype(np.int8)
    transitions = np.diff(above)
    burst_count = float(np.sum(transitions == 1))  # rising edges only
    # Normalize to [0,1] relative to max possible transitions (n_sweeps/2)
    burst_count_norm = burst_count / (n_sweeps / 2.0)

    # 4. active_channel_count: fraction of channels active at least once
    channel_totals = sweep_matrix.sum(axis=0)  # shape: (125,)
    active_channel_count = float((channel_totals > 0).sum() / NUM_CHANNELS)

    # 5. channel_hopping_rate: mean XOR distance between consecutive sweeps
    #    (fraction of channels that changed state sweep-to-sweep)
    xor_diffs = np.abs(np.diff(sweep_matrix, axis=0))  # shape: (n_sweeps-1, 125)
    channel_hopping_rate = float(xor_diffs.sum(axis=1).mean() / NUM_CHANNELS)

    # 6. peak_channel_index: index of the most active channel, normalized
    peak_ch = int(channel_totals.argmax())
    peak_channel_index = float(peak_ch / (NUM_CHANNELS - 1))

    # 7. wifi_channel_ratio: fraction of total activity in known WiFi bands
    total_activity = channel_totals.sum()
    if total_activity == 0:
        wifi_channel_ratio = 0.0
    else:
        wifi_activity = sum(
            channel_totals[lo:hi + 1].sum()
            for lo, hi in WIFI_BAND_RANGES
        )
        wifi_channel_ratio = float(wifi_activity / total_activity)

    # 8. signal_duration: fraction of sweeps with any active channel
    signal_duration = float((per_sweep_counts > 0).sum() / n_sweeps)

    return {
        "rssi_mean":            rssi_mean,
        "rssi_variance":        rssi_variance,
        "burst_count":          burst_count_norm,
        "active_channel_count": active_channel_count,
        "channel_hopping_rate": channel_hopping_rate,
        "peak_channel_index":   peak_channel_index,
        "wifi_channel_ratio":   wifi_channel_ratio,
        "signal_duration":      signal_duration,
    }


def features_to_vector(feature_dict: dict) -> np.ndarray:
    """Return features as a fixed-order numpy array (matches CSV column order)."""
    keys = [
        "rssi_mean", "rssi_variance", "burst_count", "active_channel_count",
        "channel_hopping_rate", "peak_channel_index", "wifi_channel_ratio",
        "signal_duration",
    ]
    return np.array([feature_dict[k] for k in keys], dtype=np.float32)


FEATURE_NAMES = [
    "rssi_mean", "rssi_variance", "burst_count", "active_channel_count",
    "channel_hopping_rate", "peak_channel_index", "wifi_channel_ratio",
    "signal_duration",
]


# ---------------------------------------------------------------------------
# Unit tests — run this file directly to verify feature logic
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running feature_extraction unit tests...\n")

    # Test 1: All-zero matrix → all features should be ~0
    zeros = np.zeros((30, 125), dtype=np.uint8)
    f = extract_features(zeros)
    assert f["rssi_mean"] == 0.0, "FAIL: rssi_mean on zeros"
    assert f["active_channel_count"] == 0.0, "FAIL: active_channel_count on zeros"
    assert f["signal_duration"] == 0.0, "FAIL: signal_duration on zeros"
    assert f["wifi_channel_ratio"] == 0.0, "FAIL: wifi_channel_ratio on zeros"
    print("Test 1 PASSED: All-zero matrix → all zero features")

    # Test 2: Single channel always active (channel 90, outside all WiFi bands)
    # WiFi bands occupy indices 1-23, 26-48, 51-73 — channel 90 is safe.
    single_ch = np.zeros((30, 125), dtype=np.uint8)
    single_ch[:, 90] = 1
    f = extract_features(single_ch)
    assert abs(f["rssi_mean"] - 1.0 / 125) < 1e-5, f"FAIL: rssi_mean={f['rssi_mean']}"
    assert f["rssi_variance"] == 0.0, f"FAIL: rssi_variance={f['rssi_variance']}"
    assert f["active_channel_count"] == 1.0 / 125, f"FAIL: active_channel_count"
    assert f["channel_hopping_rate"] == 0.0, "FAIL: channel_hopping_rate non-zero for static channel"
    assert f["signal_duration"] == 1.0, "FAIL: signal_duration should be 1.0"
    assert abs(f["peak_channel_index"] - 90 / 124) < 1e-5, f"FAIL: peak_channel_index={f['peak_channel_index']}"
    assert f["wifi_channel_ratio"] == 0.0, "FAIL: channel 90 should not be in WiFi band"
    print("Test 2 PASSED: Static single channel (ch 90) features correct")

    # Test 3: WiFi-like pattern — activity concentrated at ch 1 band (indices 1-23)
    wifi_sim = np.zeros((30, 125), dtype=np.uint8)
    wifi_sim[:, 1:24] = 1  # WiFi ch 1 band
    f = extract_features(wifi_sim)
    assert f["wifi_channel_ratio"] > 0.8, f"FAIL: wifi_channel_ratio={f['wifi_channel_ratio']} (expected >0.8)"
    assert f["channel_hopping_rate"] == 0.0, "FAIL: static WiFi should have zero hop rate"
    print(f"Test 3 PASSED: WiFi-band pattern → wifi_channel_ratio={f['wifi_channel_ratio']:.3f}")

    # Test 4: FHSS-like pattern — random channel changes per sweep
    # Use only channels 80-120 to stay well outside WiFi bands (1-23, 26-48, 51-73)
    rng = np.random.default_rng(42)
    fhss = np.zeros((30, 125), dtype=np.uint8)
    for i in range(30):
        hop_ch = rng.integers(80, 120)  # hop to a new channel each sweep, outside WiFi bands
        fhss[i, hop_ch:hop_ch + 2] = 1
    f = extract_features(fhss)
    assert f["channel_hopping_rate"] > 0.01, f"FAIL: FHSS hop rate too low: {f['channel_hopping_rate']}"
    assert f["wifi_channel_ratio"] == 0.0, f"FAIL: FHSS in non-WiFi bands should have zero wifi_channel_ratio, got {f['wifi_channel_ratio']}"
    print(f"Test 4 PASSED: FHSS pattern → channel_hopping_rate={f['channel_hopping_rate']:.4f}")

    # Test 5: feature_to_vector returns correct shape and order
    vec = features_to_vector(f)
    assert vec.shape == (8,), f"FAIL: vector shape={vec.shape}"
    assert vec[4] == f["channel_hopping_rate"], "FAIL: vector order mismatch at index 4"
    print("Test 5 PASSED: features_to_vector shape and order correct")

    print("\nAll tests PASSED.")
