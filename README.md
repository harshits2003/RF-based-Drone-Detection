# RF-Based Drone Detection System

A low-cost embedded system that passively detects unauthorized drones by analyzing RF signals in the 2.4 GHz band. Uses an **nRF24L01+** radio module to scan for FHSS (frequency-hopping spread spectrum) signatures and a **Random Forest** classifier exported as pure C code running on an **ESP32** — no external ML library needed.

---

## How It Works

Drones communicate with their controllers using FHSS — they hop across the 2.4 GHz band rapidly and continuously. WiFi and Bluetooth have fixed, bursty patterns. This system exploits that difference.

**Detection pipeline:**

```
nRF24L01+ RPD register
  → 30 sweeps × 125 channels (~750ms window)
  → 8 RF features (channel_hopping_rate is the primary discriminator)
  → StandardScaler normalization (baked into firmware)
  → Random Forest inference (32 trees exported as C functions)
  → 3 consecutive positive windows → GPIO 25 HIGH + Serial alert
```

**8 extracted features:**

| # | Feature | Physical meaning |
|---|---------|-----------------|
| 0 | `rssi_mean` | Sustained RF energy level |
| 1 | `rssi_variance` | Drone = steady, WiFi = bursty |
| 2 | `burst_count` | Rising-edge threshold crossings |
| 3 | `active_channel_count` | Spectral spread |
| 4 | `channel_hopping_rate` | **Primary discriminator** — FHSS drones hop fast |
| 5 | `peak_channel_index` | WiFi peaks at known indices (ch 1/6/11) |
| 6 | `wifi_channel_ratio` | Activity in WiFi ch 1/6/11 bands |
| 7 | `signal_duration` | Drone transmits continuously |

---

## Current Model Performance

Trained on 5,000 synthetic samples (2,000 drone, 3,000 non-drone):

| Metric | Value |
|--------|-------|
| Recall | **1.00** |
| False Positive Rate | 7.6% |
| F1 Score | 0.946 |
| ROC-AUC | 0.9999 |
| Decision Threshold | 0.10 (tuned for max recall) |

---

## Hardware Requirements

| Component | Notes |
|-----------|-------|
| ESP32 dev board | Any standard 38-pin variant |
| nRF24L01+**PA+LNA** | The PA+LNA variant is required for adequate range (~10–15m without it) |
| 10µF decoupling capacitor | Across VCC/GND of the nRF24L01+ — prevents brownouts |

**Default wiring (ESP32 ↔ nRF24L01+):**

| nRF24L01+ | ESP32 GPIO |
|-----------|-----------|
| CE | 4 |
| CSN | 5 |
| SCK | 18 |
| MISO | 19 |
| MOSI | 23 |
| VCC | 3.3V |
| GND | GND |

**Alert output:**
- **GPIO 25** → HIGH when drone confirmed (connect LED + resistor, or buzzer via transistor)
- Serial output at 115200 baud with per-window scores

---

## Getting Started

### Prerequisites

```bash
pip install scikit-learn numpy pandas
```

Arduino IDE with ESP32 board support. No extra libraries needed — `ml_model.h` is self-contained.

---

### Option 1 — Python pipeline only (no hardware)

Run in order to do a full rebuild from scratch:

```bash
# 1. Run feature extractor unit tests
python -X utf8 ml/feature_extraction.py

# 2. Generate synthetic training dataset
python -X utf8 dataset/generate_synthetic_dataset.py

# 3. Train the Random Forest model
python -X utf8 ml/train_model.py

# 4. Export trained model to firmware C header
python -X utf8 ml/export_to_firmware.py
```

> The models in `models/` and `firmware/ml_model.h` are pre-built — you only need to rerun this if you change the features or training data.

---

### Option 2 — Flash to ESP32

1. Open `firmware/esp32_main.ino` in Arduino IDE
2. Select your ESP32 board and COM port
3. Flash and open Serial Monitor at **115200 baud**

**Build modes** — edit the `#define` block at the top of `esp32_main.ino`:

| Mode | Define | Use case |
|------|--------|----------|
| Normal | *(commented out)* | Full operation with nRF24L01+ hardware |
| Synthetic test | `DEBUG_INJECT_SYNTHETIC` | Test firmware pipeline without RF hardware |
| Parity check | `DEBUG_PARITY` | Verify C inference matches Python output |

**Serial output example:**
```
=== RF Drone Detection System ===
Threshold: 0.1000
clear:          score=0.0312
clear:          score=0.0421
DRONE_DETECTED: score=0.8741  consecutive=1  [pending...]
DRONE_DETECTED: score=0.9102  consecutive=2  [pending...]
DRONE_DETECTED: score=0.8953  consecutive=3  [ALARM]
```

---

### Option 3 — Collect real RF data and retrain

```bash
# 1. Flash dataset/arduino_rf_scanner/rf_scanner.ino to an Arduino Uno/Nano

# 2. List available serial ports
python -X utf8 dataset/collect_real_rf_data.py --list-ports

# 3. Collect drone windows (drone must be active during capture)
python -X utf8 dataset/collect_real_rf_data.py --port COM3 --label 1 --windows 200 --out dataset/data/real_rf_features.csv

# 4. Collect no-drone baseline (appended to same file)
python -X utf8 dataset/collect_real_rf_data.py --port COM3 --label 0 --windows 200 --out dataset/data/real_rf_features.csv --append

# 5. Retrain on real data
python -X utf8 ml/train_model.py --data dataset/data/real_rf_features.csv

# 6. Re-export to firmware
python -X utf8 ml/export_to_firmware.py
```

---

## Architecture Notes

### Critical invariant

`ml/feature_extraction.py` and `firmware/feature_extraction.cpp` compute **identical features**. Any change to feature logic must be mirrored in both files. The `DEBUG_PARITY` build mode and the parity test vectors embedded in `ml_model.h` exist specifically to verify this cross-language invariant (tolerance: 0.001 absolute error).

### Dual-core ESP32 design

- **Core 0** (`loop()`): Calls `rf_collect_window()` → fills `g_sweep_buf[30][125]`
- **Core 1** (`inference_task`): Watches `g_window_ready` → `fe_extract()` → `rf_classify()` → GPIO/Serial output
- Buffer access is guarded by a FreeRTOS mutex (`g_buf_sem`)

### nRF24L01+ scanner mode

The module operates in RX mode with ShockBurst disabled. It does **not** decode packets — it only reads the RPD (Received Power Detector) register: 1-bit "signal above ~-64 dBm" per channel. Full sweep across 125 channels ≈ 25ms; 30 sweeps = ~750ms window.

### Generated files — do not hand-edit

- `firmware/ml_model.h` — regenerate via `ml/export_to_firmware.py`
- `models/drone_classifier.pkl`, `models/scaler_params.json`, `models/threshold.json` — regenerate via `ml/train_model.py`
- `dataset/data/synthetic_rf_features.csv` — regenerate via `dataset/generate_synthetic_dataset.py`

---

## Limitations

- Detection range is hardware-limited. Use nRF24L01+**PA+LNA** for anything beyond ~15m.
- Trained on synthetic data. Real-world performance improves significantly after collecting real RF samples.
- Only covers the **2.4 GHz band**. Drones using 5.8 GHz video links or 433/900 MHz control are not detected.
- Dense WiFi environments increase false positive rate.

## Future Work

- Collect and train on real-world RF captures across drone models
- Multi-node deployment for spatial localization
- SDR-based wideband detection (sub-GHz + 5.8 GHz)
- Drone model classification beyond binary detection
- OLED/web dashboard instead of Serial-only output
