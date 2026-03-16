/*
 * rf_scanner.ino — Arduino nRF24L01+ channel scanner for dataset collection.
 *
 * Sweeps all 125 channels (2.400–2.524 GHz) repeatedly and outputs
 * raw sweep data over Serial in CSV format for collection by
 * dataset/collect_real_rf_data.py.
 *
 * Wiring (Arduino Uno/Nano):
 *   Arduino  |  nRF24L01+
 *   Pin 13   →  SCK        Pin 11   →  MOSI
 *   Pin 12   →  MISO       Pin 10   →  CSN
 *   Pin 9    →  CE         3.3V     →  VCC  (NOT 5V!)
 *   GND      →  GND
 *
 * IMPORTANT: nRF24L01+ is 3.3V only. If your Arduino runs at 5V,
 * use a voltage divider or level shifter on MOSI/SCK/CSN/CE,
 * or power from the Arduino 3.3V pin (max 50mA — add capacitor).
 *
 * Serial output format (one line per sweep):
 *   SWEEP,<timestamp_ms>,<ch0>,<ch1>,...,<ch124>
 *   where each ch value is 0 or 1 (RPD register bit).
 *
 * collect_real_rf_data.py reads this format and groups sweeps
 * into observation windows before computing features.
 *
 * Baud rate: 115200
 */

#include <SPI.h>

/* ---- Pin configuration ---- */
#define PIN_CE   9
#define PIN_CSN  10

/* ---- Scanner constants ---- */
#define NUM_CHANNELS  125
#define SETTLE_US     200    /* µs per channel — must match rf_reader.h */

/* ---- nRF24L01+ registers ---- */
#define NRF_W_REG    0x20
#define NRF_R_REG    0x00
#define REG_CONFIG   0x00
#define REG_EN_AA    0x01
#define REG_RF_CH    0x05
#define REG_RF_SETUP 0x06
#define REG_RPD      0x09

static void csn_low()  { digitalWrite(PIN_CSN, LOW);  }
static void csn_high() { digitalWrite(PIN_CSN, HIGH); }
static void ce_high()  { digitalWrite(PIN_CE,  HIGH); }
static void ce_low()   { digitalWrite(PIN_CE,  LOW);  }

static void write_reg(uint8_t reg, uint8_t value) {
    csn_low();
    SPI.transfer(NRF_W_REG | (reg & 0x1F));
    SPI.transfer(value);
    csn_high();
}

static uint8_t read_reg(uint8_t reg) {
    csn_low();
    SPI.transfer(NRF_R_REG | (reg & 0x1F));
    uint8_t val = SPI.transfer(0xFF);
    csn_high();
    return val;
}

static uint8_t channel_buf[NUM_CHANNELS];

void setup() {
    Serial.begin(115200);

    pinMode(PIN_CE,  OUTPUT);
    pinMode(PIN_CSN, OUTPUT);
    ce_low();
    csn_high();

    SPI.begin();
    SPI.beginTransaction(SPISettings(8000000, MSBFIRST, SPI_MODE0));

    /* Power up in RX mode, disable CRC */
    write_reg(REG_CONFIG, 0x03);
    delayMicroseconds(1500);

    /* Disable auto-acknowledge */
    write_reg(REG_EN_AA, 0x00);

    /* 1 Mbps, 0 dBm */
    write_reg(REG_RF_SETUP, 0x06);

    /* Verify communication */
    uint8_t cfg = read_reg(REG_CONFIG);
    if (cfg == 0x00 || cfg == 0xFF) {
        Serial.println("ERROR: nRF24L01+ not detected. Check wiring.");
        for (;;);
    }

    Serial.println("# RF Scanner ready — outputting sweep CSV");
    Serial.println("# Format: SWEEP,timestamp_ms,ch0,ch1,...,ch124");
    Serial.print("# Channels: ");
    Serial.println(NUM_CHANNELS);
}

void loop() {
    unsigned long ts = millis();

    /* Print sweep header */
    Serial.print("SWEEP,");
    Serial.print(ts);

    /* Sweep all channels */
    noInterrupts();  /* disable interrupts for consistent timing */
    for (uint8_t ch = 0; ch < NUM_CHANNELS; ch++) {
        write_reg(REG_RF_CH, ch);
        ce_high();
        delayMicroseconds(SETTLE_US);
        ce_low();
        channel_buf[ch] = read_reg(REG_RPD) & 0x01;
    }
    interrupts();

    /* Output channel data */
    for (uint8_t ch = 0; ch < NUM_CHANNELS; ch++) {
        Serial.print(',');
        Serial.print(channel_buf[ch]);
    }
    Serial.println();
}
