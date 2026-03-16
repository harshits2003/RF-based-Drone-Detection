/*
 * rf_reader.h — nRF24L01+ channel scanner driver for ESP32.
 *
 * Uses the nRF24L01+ in carrier-detect (RPD register) mode to sweep all
 * 125 channels (2.400–2.524 GHz) and collect a binary activity matrix.
 *
 * Wiring (ESP32 default SPI):
 *   GPIO 18 → SCK    GPIO 19 → MISO    GPIO 23 → MOSI
 *   GPIO 5  → CSN    GPIO 4  → CE      3.3V → VCC  GND → GND
 *
 * IMPORTANT: nRF24L01+ is 3.3V only. Add a 10µF decoupling capacitor
 * on VCC as close to the module as possible.
 */

#pragma once
#include <Arduino.h>
#include <SPI.h>

/* ---- Pin configuration (override before #include if needed) ---- */
#ifndef RF_PIN_CE
  #define RF_PIN_CE   4
#endif
#ifndef RF_PIN_CSN
  #define RF_PIN_CSN  5
#endif

/* ---- Scanner constants ---- */
#define RF_NUM_CHANNELS    125    /* 2.400–2.524 GHz */
#define RF_NUM_SWEEPS       30    /* sweeps per observation window (~1 second) */

/* RPD register settle time per channel.
 * nRF24L01+ datasheet minimum: 130 µs. Use 200 µs for reliability.
 * Total sweep time ≈ 125 * 200 µs = 25 ms → 30 sweeps ≈ 750 ms window. */
#define RF_SETTLE_US       200

/* ---- nRF24L01+ SPI commands ---- */
#define NRF_W_REGISTER   0x20
#define NRF_R_REGISTER   0x00

/* ---- nRF24L01+ register addresses ---- */
#define NRF_REG_CONFIG   0x00
#define NRF_REG_EN_AA    0x01
#define NRF_REG_RF_CH    0x05
#define NRF_REG_RF_SETUP 0x06
#define NRF_REG_RPD      0x09   /* Received Power Detector — bit 0 */

/*
 * Initialize nRF24L01+ in RX/carrier-detect mode.
 * Must be called once in setup() after SPI.begin().
 *
 * Returns true if the module responds correctly (config register readable).
 */
bool rf_reader_init(uint8_t ce_pin = RF_PIN_CE, uint8_t csn_pin = RF_PIN_CSN);

/*
 * Collect a single sweep across all 125 channels.
 *
 * @param channel_activity  Output array of RF_NUM_CHANNELS bytes.
 *                          Each byte is 1 if RPD=1 on that channel, else 0.
 *
 * Sweep duration ≈ RF_NUM_CHANNELS * RF_SETTLE_US µs
 */
void rf_collect_sweep(uint8_t channel_activity[RF_NUM_CHANNELS]);

/*
 * Collect a full observation window (RF_NUM_SWEEPS sweeps).
 *
 * @param sweep_buf  Output matrix [RF_NUM_SWEEPS][RF_NUM_CHANNELS].
 *                   Caller is responsible for allocation.
 *
 * Total duration ≈ RF_NUM_SWEEPS * RF_NUM_CHANNELS * RF_SETTLE_US µs
 *                ≈ 30 * 125 * 200µs = 750 ms
 */
void rf_collect_window(uint8_t sweep_buf[RF_NUM_SWEEPS][RF_NUM_CHANNELS]);
