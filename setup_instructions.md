# Setup and Execution Instructions

This project implements a Privacy-Preserving IoT Threat Detection system using Differentially Private Federated Learning (DP-FedAvg).

## 1. Prerequisites

- **Python:** Version 3.11 or 3.12 recommended.
- **Libraries:** Install requirements using: `pip install -r requirements.txt`
- **Docker Desktop:** (Optional) Required for containerized distributed execution.

## 2. Dataset Preparation

The system uses the NSL-KDD dataset. Run this script to download, clean, and partition the data into Non-IID subsets for 3 IoT clients:

```bash
python src/prepare_dataset.py
```

## 3. Local Execution (Manual Mode)

To simulate the federated network using separate processes, open **4 different terminal windows**.

**Terminal 1: Aggregator Server**

```bash
python src/server.py 1.1
```

**Terminals 2, 3, and 4: IoT Clients (Edge Nodes)**

Wait **5 seconds** for the server to initialize, then run these in separate windows:

```bash
python src/client.py 0 1.1
python src/client.py 1 1.1
python src/client.py 2 1.1
```

## 4. Docker Execution (Distributed Mode)

To simulate a professional microservices architecture where the server and all clients run in isolated containers:

```bash
docker-compose up --build
```

## 5. Result Visualization

After the training rounds are complete, generate the analytical graphs by running:

```bash
python src/visualize_tradeoff.py
```
