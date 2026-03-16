/*
 * rf_reader.cpp — nRF24L01+ channel scanner implementation.
 *
 * The nRF24L01+ is configured in RX mode with ShockBurst disabled.
 * We iterate through all 125 channels, briefly enable CE, then read
 * the RPD (Received Power Detector) register. RPD=1 means a signal
 * above approximately -64 dBm was detected on that channel.
 *
 * References:
 *   nRF24L01+ Product Specification v1.0, Section 6.4 (Carrier Detect)
 */

#include "rf_reader.h"

/* Module-level pin storage */
static uint8_t _ce_pin  = RF_PIN_CE;
static uint8_t _csn_pin = RF_PIN_CSN;

/* ------------------------------------------------------------------ */
/* Low-level SPI helpers                                               */
/* ------------------------------------------------------------------ */

static inline void _csn_low()  { digitalWrite(_csn_pin, LOW);  }
static inline void _csn_high() { digitalWrite(_csn_pin, HIGH); }
static inline void _ce_high()  { digitalWrite(_ce_pin,  HIGH); }
static inline void _ce_low()   { digitalWrite(_ce_pin,  LOW);  }

static void _write_reg(uint8_t reg, uint8_t value) {
    _csn_low();
    SPI.transfer(NRF_W_REGISTER | (reg & 0x1F));
    SPI.transfer(value);
    _csn_high();
}

static uint8_t _read_reg(uint8_t reg) {
    _csn_low();
    SPI.transfer(NRF_R_REGISTER | (reg & 0x1F));
    uint8_t val = SPI.transfer(0xFF);
    _csn_high();
    return val;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

bool rf_reader_init(uint8_t ce_pin, uint8_t csn_pin) {
    _ce_pin  = ce_pin;
    _csn_pin = csn_pin;

    pinMode(_ce_pin,  OUTPUT);
    pinMode(_csn_pin, OUTPUT);
    _ce_low();
    _csn_high();

    SPI.begin();
    SPI.beginTransaction(SPISettings(8000000, MSBFIRST, SPI_MODE0));

    /* Power up in RX mode, disable CRC (not needed for carrier detect) */
    _write_reg(NRF_REG_CONFIG, 0x03);        /* PWR_UP=1, PRIM_RX=1 */
    delayMicroseconds(1500);                 /* tpd2stby = 1.5 ms */

    /* Disable auto-acknowledgment (ShockBurst off) */
    _write_reg(NRF_REG_EN_AA, 0x00);

    /* Set RF to 1 Mbps, 0 dBm output (doesn't affect RX sensitivity) */
    _write_reg(NRF_REG_RF_SETUP, 0x06);

    /* Sanity check: read CONFIG register back */
    uint8_t config = _read_reg(NRF_REG_CONFIG);
    if (config == 0x00 || config == 0xFF) {
        /* 0x00 or 0xFF indicates no SPI response — module not detected */
        Serial.println("[rf_reader] ERROR: nRF24L01+ not detected on SPI bus.");
        Serial.println("  Check wiring: CE=GPIO4, CSN=GPIO5, SCK=18, MISO=19, MOSI=23");
        return false;
    }
    Serial.printf("[rf_reader] Init OK — CONFIG=0x%02X\n", config);
    return true;
}

void rf_collect_sweep(uint8_t channel_activity[RF_NUM_CHANNELS]) {
    for (uint8_t ch = 0; ch < RF_NUM_CHANNELS; ch++) {
        /* Set channel */
        _write_reg(NRF_REG_RF_CH, ch);

        /* Briefly enable CE to start listening on this channel */
        _ce_high();
        delayMicroseconds(RF_SETTLE_US);
        _ce_low();

        /* Read RPD — bit 0 indicates carrier detected */
        channel_activity[ch] = _read_reg(NRF_REG_RPD) & 0x01;
    }
}

void rf_collect_window(uint8_t sweep_buf[RF_NUM_SWEEPS][RF_NUM_CHANNELS]) {
    for (int s = 0; s < RF_NUM_SWEEPS; s++) {
        rf_collect_sweep(sweep_buf[s]);
    }
}
