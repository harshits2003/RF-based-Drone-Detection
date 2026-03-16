"""
collect_real_rf_data.py — Real RF data collector for Arduino + nRF24L01+.

Reads raw sweep data from arduino_rf_scanner.ino over serial, groups
sweeps into observation windows, computes the 8 features using the
SAME feature_extraction.py logic, and saves to a labeled CSV.

Usage:
  # List available ports first:
  python dataset/collect_real_rf_data.py --list-ports

  # Collect 200 DRONE windows (fly/activate drone during capture):
  python dataset/collect_real_rf_data.py --port COM3 --label 1 \\
      --windows 200 --out dataset/data/real_rf_features.csv

  # Collect 200 NO-DRONE windows (background, WiFi, etc.):
  python dataset/collect_real_rf_data.py --port COM3 --label 0 \\
      --windows 200 --out dataset/data/real_rf_features.csv --append

  # Then retrain:
  python ml/train_model.py --data dataset/data/real_rf_features.csv

The output CSV uses EXACTLY the same schema as generate_synthetic_dataset.py,
so it can be used as a drop-in replacement for (or mixed with) synthetic data.

CSV schema:
  timestamp, rssi_mean, rssi_variance, burst_count, active_channel_count,
  channel_hopping_rate, peak_channel_index, wifi_channel_ratio,
  signal_duration, label
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

# Ensure ml/ is importable from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent / "ml"))
from feature_extraction import extract_features, FEATURE_NAMES

NUM_CHANNELS  = 125
NUM_SWEEPS    = 30   # sweeps per observation window (must match rf_reader.h RF_NUM_SWEEPS)
SWEEP_TIMEOUT = 2.0  # seconds before reporting a serial stall

CSV_COLUMNS = ["timestamp"] + FEATURE_NAMES + ["label"]


def list_ports():
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("No serial ports found.")
        for p in ports:
            print(f"  {p.device:<12}  {p.description}")
    except ImportError:
        print("pyserial not installed. Run: pip install pyserial")


def parse_sweep_line(line: str):
    """
    Parse a SWEEP CSV line from rf_scanner.ino.
    Returns (timestamp_ms, channel_array) or None on parse error.
    """
    line = line.strip()
    if not line.startswith("SWEEP,"):
        return None
    parts = line.split(",")
    if len(parts) != 2 + NUM_CHANNELS:
        return None
    try:
        ts = int(parts[1])
        channels = np.array([int(x) for x in parts[2:]], dtype=np.uint8)
        return ts, channels
    except ValueError:
        return None


def collect(port: str, baud: int, label: int, n_windows: int,
            out_path: Path, append: bool):
    try:
        import serial
    except ImportError:
        print("ERROR: pyserial not installed. Run: pip install pyserial")
        sys.exit(1)

    write_header = not (append and out_path.exists())
    mode = "a" if append else "w"

    print(f"Opening {port} at {baud} baud...")
    try:
        ser = serial.Serial(port, baud, timeout=SWEEP_TIMEOUT)
    except serial.SerialException as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Wait for Arduino reset / startup
    time.sleep(2.0)
    ser.reset_input_buffer()

    print(f"Collecting {n_windows} windows (label={label}) → {out_path}")
    if label == 1:
        print("  >>> FLY or ACTIVATE your drone controller now <<<")
    else:
        print("  >>> Keep drone OFF. Background/WiFi/BT signals only <<<")
    print()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    windows_done = 0
    sweep_buf = []      # accumulate sweeps for current window
    total_sweeps = 0
    stalled_warnings = 0

    try:
        with open(out_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if write_header:
                writer.writeheader()

            start_time = time.time()

            while windows_done < n_windows:
                try:
                    raw = ser.readline().decode("utf-8", errors="replace")
                except serial.SerialTimeoutException:
                    stalled_warnings += 1
                    if stalled_warnings <= 3:
                        print(f"  WARNING: Serial stall at sweep {total_sweeps} "
                              f"(is Arduino running rf_scanner.ino?)")
                    continue

                if raw.startswith("#"):
                    print(f"  {raw.strip()}")  # pass-through Arduino comments
                    continue

                result = parse_sweep_line(raw)
                if result is None:
                    continue

                ts_ms, channels = result
                sweep_buf.append(channels)
                total_sweeps += 1

                if len(sweep_buf) == NUM_SWEEPS:
                    # Compute features from this window
                    sweep_matrix = np.stack(sweep_buf, axis=0)  # shape: (30, 125)
                    features = extract_features(sweep_matrix)

                    row = {"timestamp": ts_ms / 1000.0}
                    row.update({k: round(v, 6) for k, v in features.items()})
                    row["label"] = label
                    writer.writerow(row)
                    f.flush()

                    windows_done += 1
                    sweep_buf = []

                    elapsed = time.time() - start_time
                    rate = windows_done / elapsed if elapsed > 0 else 0
                    eta  = (n_windows - windows_done) / rate if rate > 0 else 0
                    print(f"\r  Window {windows_done:4d}/{n_windows}  "
                          f"({rate:.1f} win/s, ETA {eta:.0f}s)", end="", flush=True)

    except KeyboardInterrupt:
        print(f"\n\nInterrupted after {windows_done} windows.")
    finally:
        ser.close()

    print(f"\n\nDone. Collected {windows_done} windows, saved to {out_path}")
    if windows_done < n_windows:
        print(f"  ({n_windows - windows_done} windows short of target)")

    _print_feature_summary(out_path, label)


def _print_feature_summary(csv_path: Path, label: int):
    """Print per-feature stats for the collected windows to sanity-check."""
    try:
        import pandas as pd
    except ImportError:
        return

    df_all = pd.read_csv(csv_path)
    df = df_all[df_all["label"] == label]
    if len(df) == 0:
        return

    print(f"\nFeature summary for label={label} ({len(df)} windows):")
    for col in FEATURE_NAMES:
        print(f"  {col:<24}  mean={df[col].mean():.4f}  std={df[col].std():.4f}  "
              f"[{df[col].min():.4f}, {df[col].max():.4f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Collect real RF data from Arduino + nRF24L01+ scanner")
    parser.add_argument("--list-ports", action="store_true",
                        help="List available serial ports and exit")
    parser.add_argument("--port",    default="COM3",
                        help="Serial port (default: COM3)")
    parser.add_argument("--baud",    type=int, default=115200,
                        help="Baud rate (default: 115200)")
    parser.add_argument("--label",   type=int, choices=[0, 1], required=False,
                        help="0=no-drone, 1=drone")
    parser.add_argument("--windows", type=int, default=100,
                        help="Number of observation windows to collect (default: 100)")
    parser.add_argument("--out",     default="dataset/data/real_rf_features.csv",
                        help="Output CSV path")
    parser.add_argument("--append",  action="store_true",
                        help="Append to existing CSV instead of overwriting")
    args = parser.parse_args()

    if args.list_ports:
        list_ports()
        return

    if args.label is None:
        parser.error("--label is required (0=no-drone, 1=drone)")

    collect(
        port=args.port,
        baud=args.baud,
        label=args.label,
        n_windows=args.windows,
        out_path=Path(args.out),
        append=args.append,
    )


if __name__ == "__main__":
    main()
