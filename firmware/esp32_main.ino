/*
 * esp32_main.ino — RF Drone Detection System main firmware
 *
 * Architecture:
 *   Core 0 (setup/loop):  RF scanning via nRF24L01+ into sweep buffer
 *   Core 1 (Task):        Feature extraction + ML inference + alert output
 *
 * Build modes (uncomment ONE):
 *   Normal operation      → (no define, both are commented out)
 *   #define DEBUG_INJECT_SYNTHETIC  → Use hardcoded test sweep matrices
 *   #define DEBUG_PARITY            → Run C/Python parity validation via serial
 *
 * Hardware:
 *   ESP32 + nRF24L01+PA+LNA module (3.3V only, add 10µF decap capacitor)
 *   GPIO 25: LED output (HIGH = drone detected)
 *   GPIO 26: Serial output for monitoring
 *
 * Libraries required (Arduino Library Manager):
 *   None for inference (ml_model.h is self-contained)
 *   SPI (built-in)
 */

#include <Arduino.h>
#include <SPI.h>
#include "rf_reader.h"
#include "feature_extraction.h"
#include "ml_model.h"

/* ---- Build mode selection ---------------------------------------- */
// #define DEBUG_INJECT_SYNTHETIC
// #define DEBUG_PARITY

/* ---- Output pin configuration ------------------------------------ */
#define PIN_DRONE_LED   25   /* HIGH when drone detected             */
#define PIN_CONF_OUT    26   /* Analog confidence (future DAC use)   */

/* ---- Detection persistence (consecutive window filter) ------------ */
#define DRONE_PERSIST_COUNT  3   /* consecutive positive windows to confirm */

/* ---- Global state ------------------------------------------------- */
static uint8_t  g_sweep_buf[RF_NUM_SWEEPS][RF_NUM_CHANNELS];
static float    g_features[FE_N_FEATURES];
static bool     g_window_ready  = false;
static bool     g_inference_done = true;
static SemaphoreHandle_t g_buf_sem;   /* mutex for sweep buffer */

static int  g_consecutive_detections = 0;
static bool g_alarm_active = false;

/* ================================================================
 * DEBUG: Synthetic test sweep matrices
 * ================================================================ */
#ifdef DEBUG_INJECT_SYNTHETIC

/* 5 synthetic sweep matrices: first 3 are no-drone, last 2 are drone-like.
 * Each is 30 sweeps × 125 channels stored as compressed active-channel lists.
 * Expanded at runtime via inject_synthetic_sweep(). */

typedef struct {
    uint8_t label;          /* 0=no-drone, 1=drone */
    const char* desc;
    uint8_t active_ch;      /* single representative active channel per sweep */
    uint8_t hop_range_lo;   /* FHSS: lower channel bound */
    uint8_t hop_range_hi;   /* FHSS: upper channel bound */
} SyntheticPattern;

static const SyntheticPattern SYNTHETIC_PATTERNS[] = {
    {0, "Background noise",  0,   0,   0},   /* all zeros */
    {0, "WiFi ch 1",         12,  1,  23},   /* static WiFi ch 1 band */
    {0, "WiFi ch 6",         37, 26,  48},   /* static WiFi ch 6 band */
    {1, "Drone FHSS fast",    0,  80, 120},  /* fast hopping outside WiFi */
    {1, "Drone FHSS medium",  0,  75, 110},  /* medium hopping */
};

static void fill_synthetic(int pattern_idx) {
    const SyntheticPattern& p = SYNTHETIC_PATTERNS[pattern_idx];
    uint32_t seed = (uint32_t)pattern_idx * 12345 + 99;

    for (int s = 0; s < RF_NUM_SWEEPS; s++) {
        memset(g_sweep_buf[s], 0, RF_NUM_CHANNELS);
        if (p.label == 0 && p.active_ch > 0) {
            /* Static WiFi: fill the band */
            for (int c = p.hop_range_lo; c <= p.hop_range_hi && c < RF_NUM_CHANNELS; c++) {
                g_sweep_buf[s][c] = 1;
            }
        } else if (p.label == 1) {
            /* FHSS drone: hop to random channel within range each sweep */
            seed = seed * 1664525 + 1013904223;  /* LCG */
            uint8_t hop = p.hop_range_lo + (seed % (p.hop_range_hi - p.hop_range_lo + 1));
            if (hop < RF_NUM_CHANNELS)     g_sweep_buf[s][hop]     = 1;
            if (hop + 1 < RF_NUM_CHANNELS) g_sweep_buf[s][hop + 1] = 1;
        }
        /* background: all zeros (memset above handles it) */
    }
}

#endif /* DEBUG_INJECT_SYNTHETIC */

/* ================================================================
 * Core 1 Task: Inference
 * ================================================================ */

static void inference_task(void* param) {
    for (;;) {
        /* Wait until a window is ready */
        if (!g_window_ready) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        /* Take mutex to safely copy the buffer */
        if (xSemaphoreTake(g_buf_sem, pdMS_TO_TICKS(100)) == pdTRUE) {
            /* Feature extraction */
            fe_extract((const uint8_t (*)[RF_NUM_CHANNELS])g_sweep_buf,
                       RF_NUM_SWEEPS, RF_NUM_CHANNELS, g_features);
            g_window_ready  = false;
            g_inference_done = false;
            xSemaphoreGive(g_buf_sem);
        }

        /* Normalize + classify */
        float features_copy[FE_N_FEATURES];
        memcpy(features_copy, g_features, sizeof(g_features));
        bool drone_detected = rf_classify(features_copy);   /* from ml_model.h */
        float prob = 0.0f;
        {
            float f2[FE_N_FEATURES];
            memcpy(f2, g_features, sizeof(g_features));
            normalize_features(f2);
            prob = rf_predict_proba(f2);
        }

        /* Persistence filter: require DRONE_PERSIST_COUNT consecutive positives */
        if (drone_detected) {
            g_consecutive_detections++;
        } else {
            g_consecutive_detections = 0;
        }
        bool confirmed = (g_consecutive_detections >= DRONE_PERSIST_COUNT);

        /* Update alarm state and outputs */
        if (confirmed && !g_alarm_active) {
            g_alarm_active = true;
            digitalWrite(PIN_DRONE_LED, HIGH);
        } else if (!confirmed && g_alarm_active && g_consecutive_detections == 0) {
            g_alarm_active = false;
            digitalWrite(PIN_DRONE_LED, LOW);
        }

        /* Serial output */
        if (drone_detected) {
            Serial.printf("DRONE_DETECTED: score=%.4f  consecutive=%d%s\n",
                          prob, g_consecutive_detections,
                          confirmed ? "  [ALARM]" : "  [pending...]");
        } else {
            Serial.printf("clear:          score=%.4f\n", prob);
        }

#ifdef DEBUG_INJECT_SYNTHETIC
        fe_print(g_features);
#endif

        g_inference_done = true;
    }
}

/* ================================================================
 * DEBUG_PARITY: C/Python parity validation
 * ================================================================ */

#ifdef DEBUG_PARITY
static void run_parity_test() {
    Serial.println("\n=== PARITY TEST (compare with python export_to_firmware.py output) ===");
    Serial.printf("  #   prob_c   pred_c   expected_prob  match\n");

    int pass = 0, fail = 0;
    for (int i = 0; i < PARITY_N_SAMPLES; i++) {
        /* Apply scaler to raw features */
        float f[N_FEATURES];
        for (int j = 0; j < N_FEATURES; j++) {
            f[j] = (PARITY_RAW_FEATURES[i][j] - FEATURE_MEAN[j]) / FEATURE_STD[j];
        }
        float prob = rf_predict_proba(f);
        uint8_t pred = (prob >= DECISION_THRESHOLD) ? 1 : 0;
        bool match = (fabsf(prob - PARITY_EXPECTED_PROBA[i]) < 0.002f);
        Serial.printf("  %2d  %.4f   %d        %.4f         %s\n",
                      i, prob, pred, PARITY_EXPECTED_PROBA[i],
                      match ? "OK" : "FAIL");
        match ? pass++ : fail++;
    }
    Serial.printf("\nResult: %d PASS, %d FAIL (tolerance 0.002)\n", pass, fail);
    if (fail == 0) Serial.println(">>> ALL PARITY TESTS PASSED <<<");
    else           Serial.println(">>> PARITY FAILURES DETECTED — retrain and re-export <<<");
}
#endif

/* ================================================================
 * setup() and loop()
 * ================================================================ */

void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n=== RF Drone Detection System ===");
    Serial.printf("Threshold: %.4f\n", DECISION_THRESHOLD);

    /* Output pins */
    pinMode(PIN_DRONE_LED, OUTPUT);
    digitalWrite(PIN_DRONE_LED, LOW);

#ifdef DEBUG_PARITY
    /* Run parity test immediately, then halt */
    run_parity_test();
    Serial.println("Parity test complete. Halting (comment out DEBUG_PARITY to run normally).");
    for (;;) delay(1000);
#endif

#ifndef DEBUG_INJECT_SYNTHETIC
    /* Initialize nRF24L01+ */
    if (!rf_reader_init(RF_PIN_CE, RF_PIN_CSN)) {
        Serial.println("Halting: nRF24L01+ init failed. Check wiring.");
        for (;;) delay(1000);
    }
#else
    Serial.println("DEBUG_INJECT_SYNTHETIC mode — no RF hardware required.");
#endif

    /* Create buffer mutex */
    g_buf_sem = xSemaphoreCreateMutex();

    /* Launch inference task on Core 1 */
    xTaskCreatePinnedToCore(
        inference_task,
        "inference",
        8192,   /* stack size in bytes */
        NULL,
        1,      /* priority */
        NULL,
        1       /* Core 1 */
    );

    Serial.println("Setup complete. Starting RF scan loop...\n");
}

static int _inject_idx = 0;  /* for cycling through synthetic patterns */

void loop() {
    /* Wait until previous inference is done to avoid buffer race */
    while (!g_inference_done) {
        delay(5);
    }

    /* Collect (or inject) sweep window into g_sweep_buf */
    if (xSemaphoreTake(g_buf_sem, pdMS_TO_TICKS(200)) == pdTRUE) {

#ifdef DEBUG_INJECT_SYNTHETIC
        int pattern_count = sizeof(SYNTHETIC_PATTERNS) / sizeof(SYNTHETIC_PATTERNS[0]);
        fill_synthetic(_inject_idx);
        Serial.printf("[inject] Pattern %d: %s (label=%d)\n",
                      _inject_idx,
                      SYNTHETIC_PATTERNS[_inject_idx].desc,
                      SYNTHETIC_PATTERNS[_inject_idx].label);
        _inject_idx = (_inject_idx + 1) % pattern_count;
        delay(500);   /* slow down for readability */
#else
        rf_collect_window(g_sweep_buf);
#endif

        g_window_ready   = true;
        g_inference_done = false;
        xSemaphoreGive(g_buf_sem);
    }
}
