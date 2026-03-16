/*
 * feature_extraction.h — C port of ml/feature_extraction.py
 *
 * Computes 8 RF signal features from a binary sweep matrix collected by
 * rf_reader.cpp. Must produce numerically identical results to the Python
 * implementation (validated to within 0.001 float32 tolerance).
 *
 * Feature order matches FEATURE_MEAN/FEATURE_STD arrays in ml_model.h:
 *   0: rssi_mean             4: channel_hopping_rate
 *   1: rssi_variance         5: peak_channel_index
 *   2: burst_count           6: wifi_channel_ratio
 *   3: active_channel_count  7: signal_duration
 */

#pragma once
#include <stdint.h>
#include "rf_reader.h"   /* for RF_NUM_SWEEPS, RF_NUM_CHANNELS */

#define FE_N_FEATURES      8
#define FE_BURST_THRESHOLD 2   /* must match Python BURST_THRESHOLD */

/* WiFi 2.4 GHz band index ranges (inclusive):
 *   Ch 1  (2.412 GHz, idx 12) ± 11 MHz → indices  1–23
 *   Ch 6  (2.437 GHz, idx 37) ± 11 MHz → indices 26–48
 *   Ch 11 (2.462 GHz, idx 62) ± 11 MHz → indices 51–73
 *
 * These must exactly match WIFI_BAND_RANGES in ml/feature_extraction.py.
 */
#define FE_WIFI_N_BANDS 3
static const uint8_t FE_WIFI_BANDS[FE_WIFI_N_BANDS][2] = {
    { 1,  23},   /* WiFi ch 1  */
    {26,  48},   /* WiFi ch 6  */
    {51,  73},   /* WiFi ch 11 */
};

/*
 * Compute all 8 features from a sweep matrix.
 *
 * @param sweep_buf   Input matrix [n_sweeps][n_channels], uint8_t (0 or 1)
 * @param n_sweeps    Number of sweeps (rows)
 * @param n_channels  Number of channels (cols) — must be RF_NUM_CHANNELS = 125
 * @param features    Output float array of FE_N_FEATURES values (pre-allocated)
 *
 * Features are written in this order:
 *   features[0] = rssi_mean
 *   features[1] = rssi_variance
 *   features[2] = burst_count
 *   features[3] = active_channel_count
 *   features[4] = channel_hopping_rate
 *   features[5] = peak_channel_index
 *   features[6] = wifi_channel_ratio
 *   features[7] = signal_duration
 */
void fe_extract(const uint8_t sweep_buf[][RF_NUM_CHANNELS],
                int n_sweeps, int n_channels,
                float features[FE_N_FEATURES]);

/*
 * Print feature vector to Serial for debugging.
 */
void fe_print(const float features[FE_N_FEATURES]);
