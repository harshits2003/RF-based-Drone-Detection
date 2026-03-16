# RF-Based Unauthorized Drone Detection System
> **Using Feature-Based Signal Analysis and TinyML**

---

## 🎯 Project Vision
To design a low-cost embedded system capable of detecting unauthorized drones by analyzing RF communication signals between drones and their controllers. This system utilizes lightweight machine learning (TinyML) deployed directly on an **ESP32 microcontroller**.

## 💡 Motivation
The proliferation of drones in civilian and military sectors introduces significant security risks, including:
* **Surveillance** of restricted areas.
* **Smuggling** across borders.
* **Intrusions** into airports or military bases.
* **Espionage** near government buildings.

While traditional systems (radar, cameras, acoustic) exist, they are often cost-prohibitive. RF-based detection offers a **passive, cost-effective alternative**.

---

## ❓ Problem Statement
Unauthorized drones in restricted airspace threaten critical infrastructure. There is an urgent need for a **compact, embedded system** that monitors RF activity and accurately identifies drone communication signatures in real-time.

---

## 🚀 Objectives

### Primary Objectives
* **Monitor:** Scan RF spectrum for drone communication signals.
* **Extract:** Identify key signal features (RSSI, burst count, etc.).
* **Train:** Develop a lightweight ML model.
* **Deploy:** Implement the model on ESP32 using **TinyML**.
* **Alert:** Generate real-time notifications upon detection.

### Secondary Objectives
* Minimize False Positives (distinguishing from WiFi).
* Maintain low power consumption for portable use.
* Keep hardware costs accessible.

---

## 🏗️ System Architecture



**Workflow:**
1.  **RF Environment:** Capture signals via 2.4 GHz antenna.
2.  **Front-End:** RF receiver (CC2500/nRF24L01) measures characteristics.
3.  **Feature Extraction:** Process raw signal data into feature vectors.
4.  **Inference:** TinyML model on ESP32 classifies the signal.
5.  **Action:** Trigger LED, buzzer, or serial alert if a drone is detected.

---

## 🛠️ Hardware & Software Specs

### Hardware Components
| Component | Details |
| :--- | :--- |
| **Microcontroller** | **ESP32** (Dual-core, WiFi/BT, TinyML ready) |
| **RF Module** | **CC2500** (Recommended) or nRF24L01+ |
| **Antenna** | 2.4 GHz Dipole / Omnidirectional |
| **Alerts** | LED, Buzzer, Serial Output |

### Software Stack
* **Embedded:** C/C++ (Arduino Framework), TensorFlow Lite Micro.
* **ML Training:** Python, Scikit-learn, TensorFlow.
* **Frequency Target:** Primary focus on the **2.4 GHz band**.

---

## 📊 Machine Learning Pipeline

### RF Signal Features
The model analyzes the following vector:
`[ RSSI_mean, RSSI_variance, burst_count, packet_rate, signal_duration ]`

### TinyML Deployment Workflow
1.  **Train:** Python Model (Decision Tree/Random Forest).
2.  **Convert:** TensorFlow Lite → TensorFlow Lite Micro.
3.  **Encode:** Convert to **C Array**.
4.  **Flash:** Deploy to ESP32 Firmware.

---

## 📈 Performance Targets
* **Detection Accuracy:** >85%
* **False Positives:** <10%
* **Inference Time:** <100 ms
* **Detection Range:** 100–300 meters

---

## ⚠️ Limitations & Future Scope
* **Current Limitations:** Distinguishing between high-traffic WiFi and drones; limited scanning bandwidth.
* **Future Improvements:** Drone type classification, multi-node localization, and SDR-based wideband detection.

---



## 📉 Performance Targets
| Metric | Target Goal |
| :--- | :--- |
| **False Positives** | < 10% |
| **Inference Time** | < 100 ms |
| **Detection Range** | 100–300 meters |

---

## ⚠️ System Limitations
Despite the robust design, certain technical challenges exist:
* **Signal Overlap:** Difficulty distinguishing between high-traffic WiFi and drone control signals.
* **Scanning Bandwidth:** Limited instantaneous RF bandwidth scanning on low-cost modules.
* **Environmental Noise:** Significant RF interference in urban or industrial zones.
* **Range:** Physical limitations of the 2.4 GHz receiver sensitivity.

### 🛠️ Mitigation Strategies
* **Advanced Engineering:** Implementation of more granular feature engineering.
* **Data Diversity:** Training on a larger, more diverse dataset including various WiFi signatures.
* **Hardware Upgrades:** Utilizing improved high-gain antennas.

---

## 🔮 Future Improvements
* **SDR Integration:** High-resolution detection using Software Defined Radio.
* **Classification:** Identifying specific drone models or types.
* **Localization:** Using a multi-node network and GPS for drone tracking.
* **Defense:** Integration with active anti-drone/jamming systems.
* **Live Analysis:** Real-time wide-spectrum waterfall analysis.

---

## 🏢 Applications
* **Military:** Base protection and border surveillance.
* **Civilian:** Airport airspace monitoring, event security, and prison contraband prevention.
* **Industrial:** Critical infrastructure protection and smart city security monitoring.

---

## ✅ Expected Project Outcome
The final prototype will serve as a **Proof of Concept (PoC)** demonstrating:
1.  Continuous RF signal monitoring.
2.  Automated drone signal identification.
3.  Edge-based ML classification (TinyML).
4.  Low-latency real-time alerting.

The system will prove that **Embedded AI + RF Analysis** provides a viable, low-cost solution for modern security challenges.

---

## 📁 Repository Structure
```text
RF_Drone_Detection/
├── hardware/
│   ├── circuit_diagram/
│   └── antenna_setup/
├── firmware/
│   ├── esp32_main.ino
│   ├── rf_reader.cpp
│   ├── feature_extraction.cpp
│   └── ml_inference.cpp
├── ml_training/
│   ├── dataset_collection.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   └── convert_to_tflite.py
├── models/
│   └── drone_classifier.tflite
├── data/
│   ├── raw_rf_samples/
│   └── processed_features/
└── docs/
    ├── project_report/
    └── literature_review/