# Privacy-Preserving Threat Detection in IoT Using Differentially Private FedAvg

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Privacy](https://img.shields.io/badge/Security-Differential%20Privacy-green)
![Federated Learning](https://img.shields.io/badge/Federated%20Learning-Flower-red)

## 📋 Project Overview

This final year capstone project implements a **Privacy-Preserving Intrusion Detection System (IDS)** specifically designed for IoT ecosystems.

IoT devices generate highly sensitive network logs, which are protected under laws like **GDPR**. Traditionally, training AI requires centralizing this data, risking massive privacy breaches. Our system solves this by using **Federated Learning (FedAvg)** to keep data local and **Differential Privacy (DP)** to mask model updates, ensuring that sensitive information can never be reverse-engineered from the shared intelligence.

## 🚀 Key Features

- **Decentralized Training:** Local IoT nodes train models on-device; raw logs never leave the edge.
- **Manual DP-SGD Implementation:** Custom gradient clipping and Gaussian noise injection for granular privacy control.
- **Realistic IoT Simulation:** Uses **Non-IID** data partitioning to simulate unique traffic patterns across diverse devices.
- **Mathematical Privacy Proof:** Real-time tracking of the **Epsilon ($\epsilon$)** privacy budget.
- **Privacy-Utility Analysis:** Extensive benchmarking to find the optimal balance between security and detection accuracy.

## 🏆 Results Summary

We evaluated the system across 5 different noise levels to measure the trade-off between privacy and accuracy.

| Noise Level          | Privacy Budget (ε)    | Detection Accuracy | F1-Score | Status              |
| :------------------- | :-------------------- | :----------------- | :------- | :------------------ |
| **0.0 (Baseline)**   | Infinite (No Privacy) | **80.11%**         | 0.792    | Maximum Performance |
| **0.5 (Low Noise)**  | 142.32                | 75.31%             | 0.728    | High Leakage        |
| **1.1 (Standard)**   | 64.69                 | 79.33%             | 0.793    | Balanced            |
| **2.0 (High Noise)** | 35.58                 | 76.87%             | 0.761    | Strong Privacy      |
| **4.0 (Max Noise)**  | **17.79 (Strongest)** | **79.07%**         | 0.789    | **Ideal Security**  |

### Key Observations

- **Robustness:** Even at maximum privacy (Noise 4.0), accuracy dropped by only **~1%** compared to the baseline.
- **Resilience:** The system successfully built a smart global model even with imbalanced (Non-IID) data distributions.

## 🧠 Methodology

1. **Data Engineering:** Pre-processing the **NSL-KDD** dataset and partitioning it into Non-IID client subsets.
2. **Local Intelligence:** Building a 4-layer **Deep Neural Network (DNN)** for binary threat classification at the edge.
3. **Privacy Layer:** Intercepting gradients via **TensorFlow GradientTape** to apply clipping and Gaussian noise.
4. **Federated Aggregation:** Using the **Flower (flwr)** framework to perform secure weight averaging at the server.
5. **Global Evaluation:** Testing the aggregated model against a separate, unseen global test set.

## 📄 Dataset Information

- **Dataset:** NSL-KDD (IoT Security Logs)
- **Total Records:** 148,517
- **Features:** 41 (Network traffic attributes)
- **Classification:** Binary (Normal vs. Attack)
- **Distribution:** Non-IID (Imbalanced across 3 IoT nodes)

## 🔮 Future Work

- **Docker Containerization:** Transitioning to a distributed microservices architecture to simulate independent hardware environments.
- **Scalability Testing:** Increasing the number of IoT clients to evaluate global model stability in larger networks.
- **Attack Granularity:** Implementing Confusion Matrices to analyze detection rates for specific attack types (DoS, Probe, R2L, U2R).
