# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RF-based drone detection system using nRF24L01+ channel scanning + TinyML on ESP32. The system detects drones by recognizing their FHSS (frequency-hopping spread spectrum) RF signature across the 2.4 GHz band.

**Two-phase workflow:**
1. Python pipeline: generate dataset → train model → export to C header
2. ESP32 firmware: scan RF → extract features → run embedded classifier → alert

---

## Commands

### Python pipeline (run in order for a full rebuild)

```bash
# 1. Run feature extractor unit tests
python -X utf8 ml/feature_extraction.py

# 2. Generate synthetic training dataset (5000 samples)
python -X utf8 dataset/generate_synthetic_dataset.py

# 3. Train model and save artifacts to models/
python -X utf8 ml/train_model.py

# 4. Export trained model to firmware/ml_model.h
python -X utf8 ml/export_to_firmware.py

# Optional: train on real RF data instead of (or mixed with) synthetic
python -X utf8 ml/train_model.py --data dataset/data/real_rf_features.csv
```

### Real RF data collection (requires Arduino + nRF24L01+)

```bash
# List serial ports
python -X utf8 dataset/collect_real_rf_data.py --list-ports

# Collect drone windows (drone active during capture)
python -X utf8 dataset/collect_real_rf_data.py --port COM3 --label 1 --windows 200 --out dataset/data/real_rf_features.csv

# Collect no-drone windows (append to same file)
python -X utf8 dataset/collect_real_rf_data.py --port COM3 --label 0 --windows 200 --out dataset/data/real_rf_features.csv --append
```

### ESP32 firmware

Open `firmware/esp32_main.ino` in Arduino IDE. Required libraries: SPI (built-in). `ml_model.h` is self-contained — no extra libraries needed for inference.

**Build modes** — uncomment one `#define` at the top of `esp32_main.ino`:
- *(no define)*: Normal operation with nRF24L01+ hardware
- `DEBUG_INJECT_SYNTHETIC`: Test full firmware pipeline without RF hardware
- `DEBUG_PARITY`: Verify C inference matches Python output (compare serial to `export_to_firmware.py` parity table)

---

## Architecture

### Critical invariant: `ml/feature_extraction.py` ↔ `firmware/feature_extraction.cpp`

These two files compute identical features. **Any change to feature logic must be mirrored in both files.** The 8 features (in fixed order) are:

| Index | Feature | Key physical intuition |
|-------|---------|----------------------|
| 0 | `rssi_mean` | Sustained RF energy |
| 1 | `rssi_variance` | Drone=steady, WiFi=bursty |
| 2 | `burst_count` | Rising-edge threshold crossings |
| 3 | `active_channel_count` | Channel spread |
| 4 | `channel_hopping_rate` | **Primary discriminator** — FHSS drone hops fast |
| 5 | `peak_channel_index` | WiFi peaks at known indices |
| 6 | `wifi_channel_ratio` | Activity in WiFi ch 1/6/11 bands |
| 7 | `signal_duration` | Drone transmits continuously |

WiFi band index ranges (indices 0–124 = 2.400–2.524 GHz) — defined in **both** files and must stay in sync:
- Ch 1: indices 1–23, Ch 6: indices 26–48, Ch 11: indices 51–73

### Data flow

```
nRF24L01+ RPD register
  → uint8_t[30 sweeps][125 channels]   (rf_reader.cpp collects ~750ms)
  → float[8 features]                  (feature_extraction.cpp)
  → normalize_features()               (ml_model.h: subtract FEATURE_MEAN, divide FEATURE_STD)
  → rf_predict_proba()                 (ml_model.h: Random Forest trees in C)
  → score >= DECISION_THRESHOLD        (ml_model.h: currently 0.10)
  → 3 consecutive positives            (esp32_main.ino: persistence filter)
  → GPIO 25 HIGH + Serial alert
```

### Dataset CSV schema

All CSV files (synthetic and real) share this exact column order — required by `train_model.py`:
```
timestamp, rssi_mean, rssi_variance, burst_count, active_channel_count,
channel_hopping_rate, peak_channel_index, wifi_channel_ratio, signal_duration, label
```
`label`: 1=drone, 0=no-drone. Real data from `collect_real_rf_data.py` drops into this schema directly.

### Generated files — do not hand-edit

- `firmware/ml_model.h` — always regenerate via `ml/export_to_firmware.py`
- `models/drone_classifier.pkl`, `models/scaler_params.json`, `models/threshold.json` — always regenerate via `ml/train_model.py`
- `dataset/data/synthetic_rf_features.csv` — regenerate via `dataset/generate_synthetic_dataset.py`

### ESP32 dual-core design

- **Core 0** (`loop()`): Calls `rf_collect_window()` to fill `g_sweep_buf`, sets `g_window_ready = true`
- **Core 1** (`inference_task`): Watches `g_window_ready`, runs `fe_extract()` → `rf_classify()`, updates GPIO/Serial
- Buffer access is guarded by `g_buf_sem` (FreeRTOS mutex)

### nRF24L01+ scanner mode

The module is configured in RX mode with ShockBurst disabled. It does **not** decode packets — it only uses the RPD (Received Power Detector) register, which returns 1-bit "signal above ~-64 dBm" per channel. Sweep: iterate channels 0–124, pulse CE high for 200µs per channel, read RPD. Full sweep ≈ 25ms; 30 sweeps = ~750ms window.

**Important:** Use the `nRF24L01+PA+LNA` variant for adequate range. The plain nRF24L01+ RPD threshold (~-64 dBm) limits detection range to ~10–15m.

### Model export and scaler constants

`export_to_firmware.py` generates `ml_model.h` containing:
1. `FEATURE_MEAN[8]` and `FEATURE_STD[8]` — baked-in StandardScaler constants
2. `DECISION_THRESHOLD` — tuned on validation set (not 0.5)
3. `_tree_N()` functions — one C function per decision tree in the Random Forest
4. `rf_predict_proba(float* f)` — averages all tree outputs
5. `rf_classify(float* f)` — normalize + predict convenience wrapper
6. Parity test vectors for `DEBUG_PARITY` validation

The threshold is tuned to maximize recall while keeping FPR < 10% on the validation set.
