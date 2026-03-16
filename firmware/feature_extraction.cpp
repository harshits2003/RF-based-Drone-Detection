/*
 * feature_extraction.cpp — C port of ml/feature_extraction.py
 *
 * All computations mirror the Python implementation exactly.
 * Parity tolerance target: 0.001 absolute error per feature
 * on identical input matrices.
 *
 * Numerical notes:
 *   - All intermediate sums use float32 (matches numpy float32)
 *   - Variance uses two-pass algorithm (same as numpy std())
 *   - XOR differences use integer subtraction on 0/1 values (same as numpy abs diff)
 */

#include "feature_extraction.h"
#include <math.h>   /* sqrtf */
#include <Arduino.h>

void fe_extract(const uint8_t sweep_buf[][RF_NUM_CHANNELS],
                int n_sweeps, int n_channels,
                float features[FE_N_FEATURES])
{
    /* ----------------------------------------------------------
     * Step 1: Per-sweep active channel counts
     * per_sweep_counts[s] = number of active channels in sweep s
     * ---------------------------------------------------------- */
    float per_sweep_counts[RF_NUM_SWEEPS];
    for (int s = 0; s < n_sweeps; s++) {
        int cnt = 0;
        for (int c = 0; c < n_channels; c++) {
            cnt += sweep_buf[s][c];
        }
        per_sweep_counts[s] = (float)cnt;
    }

    /* ----------------------------------------------------------
     * Feature 0: rssi_mean = mean(per_sweep_counts) / n_channels
     * ---------------------------------------------------------- */
    float sum = 0.0f;
    for (int s = 0; s < n_sweeps; s++) sum += per_sweep_counts[s];
    float mean_counts = sum / (float)n_sweeps;
    features[0] = mean_counts / (float)n_channels;

    /* ----------------------------------------------------------
     * Feature 1: rssi_variance = std(per_sweep_counts) / n_channels
     * Two-pass variance: sum((x - mean)^2) / n (population std)
     * Matches numpy's default std(ddof=0)
     * ---------------------------------------------------------- */
    float var_sum = 0.0f;
    for (int s = 0; s < n_sweeps; s++) {
        float diff = per_sweep_counts[s] - mean_counts;
        var_sum += diff * diff;
    }
    float std_counts = sqrtf(var_sum / (float)n_sweeps);
    features[1] = std_counts / (float)n_channels;

    /* ----------------------------------------------------------
     * Feature 2: burst_count
     * Count rising-edge threshold crossings, normalized by (n_sweeps/2).
     * A rising edge = per_sweep_counts[s-1] <= threshold AND
     *                 per_sweep_counts[s]   >  threshold
     * Matches Python: np.diff((counts > THRESHOLD).astype(int)) == 1
     * ---------------------------------------------------------- */
    int burst_count = 0;
    int prev_above = (per_sweep_counts[0] > (float)FE_BURST_THRESHOLD) ? 1 : 0;
    for (int s = 1; s < n_sweeps; s++) {
        int curr_above = (per_sweep_counts[s] > (float)FE_BURST_THRESHOLD) ? 1 : 0;
        if (curr_above == 1 && prev_above == 0) {
            burst_count++;
        }
        prev_above = curr_above;
    }
    features[2] = (float)burst_count / ((float)n_sweeps / 2.0f);

    /* ----------------------------------------------------------
     * Feature 3: active_channel_count
     * Fraction of channels active (RPD=1) at least once.
     * channel_totals[c] = sum over all sweeps of sweep_buf[s][c]
     * ---------------------------------------------------------- */
    uint16_t channel_totals[RF_NUM_CHANNELS];
    for (int c = 0; c < n_channels; c++) {
        uint16_t tot = 0;
        for (int s = 0; s < n_sweeps; s++) {
            tot += sweep_buf[s][c];
        }
        channel_totals[c] = tot;
    }
    int active_ch = 0;
    for (int c = 0; c < n_channels; c++) {
        if (channel_totals[c] > 0) active_ch++;
    }
    features[3] = (float)active_ch / (float)n_channels;

    /* ----------------------------------------------------------
     * Feature 4: channel_hopping_rate
     * Mean XOR distance between consecutive sweeps / n_channels.
     * XOR distance = number of channels that changed state.
     * For 0/1 arrays: |a - b| == a XOR b, so sum of |diff| = XOR count.
     * ---------------------------------------------------------- */
    float xor_sum = 0.0f;
    for (int s = 1; s < n_sweeps; s++) {
        int row_xor = 0;
        for (int c = 0; c < n_channels; c++) {
            row_xor += (sweep_buf[s][c] != sweep_buf[s-1][c]) ? 1 : 0;
        }
        xor_sum += (float)row_xor;
    }
    features[4] = (xor_sum / (float)(n_sweeps - 1)) / (float)n_channels;

    /* ----------------------------------------------------------
     * Feature 5: peak_channel_index
     * Index of channel with highest total activity / (n_channels - 1)
     * ---------------------------------------------------------- */
    int peak_ch = 0;
    uint16_t peak_val = channel_totals[0];
    for (int c = 1; c < n_channels; c++) {
        if (channel_totals[c] > peak_val) {
            peak_val = channel_totals[c];
            peak_ch  = c;
        }
    }
    features[5] = (float)peak_ch / (float)(n_channels - 1);

    /* ----------------------------------------------------------
     * Feature 6: wifi_channel_ratio
     * Fraction of total activity in known WiFi band index ranges.
     * ---------------------------------------------------------- */
    uint32_t total_activity = 0;
    for (int c = 0; c < n_channels; c++) total_activity += channel_totals[c];

    if (total_activity == 0) {
        features[6] = 0.0f;
    } else {
        uint32_t wifi_activity = 0;
        for (int b = 0; b < FE_WIFI_N_BANDS; b++) {
            uint8_t lo = FE_WIFI_BANDS[b][0];
            uint8_t hi = FE_WIFI_BANDS[b][1];
            for (int c = lo; c <= hi; c++) {
                wifi_activity += channel_totals[c];
            }
        }
        features[6] = (float)wifi_activity / (float)total_activity;
    }

    /* ----------------------------------------------------------
     * Feature 7: signal_duration
     * Fraction of sweeps with any active channel (count > 0)
     * ---------------------------------------------------------- */
    int active_sweeps = 0;
    for (int s = 0; s < n_sweeps; s++) {
        if (per_sweep_counts[s] > 0.0f) active_sweeps++;
    }
    features[7] = (float)active_sweeps / (float)n_sweeps;
}

void fe_print(const float features[FE_N_FEATURES]) {
    const char* names[FE_N_FEATURES] = {
        "rssi_mean", "rssi_variance", "burst_count", "active_channel_count",
        "channel_hopping_rate", "peak_channel_index", "wifi_channel_ratio",
        "signal_duration"
    };
    Serial.println("[features]");
    for (int i = 0; i < FE_N_FEATURES; i++) {
        Serial.printf("  %-24s = %.5f\n", names[i], features[i]);
    }
}
