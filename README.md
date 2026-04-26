# Real-Time ML Feature Store & Model Serving Platform

An end-to-end MLOps pipeline that streams credit card transactions through Kafka,
engineers features with PySpark, stores them in Redis, and serves real-time fraud
predictions via a FastAPI + MLflow API — with drift monitoring built in.

---

## Architecture

```
                        ┌─────────────────────────────────────────────────┐
                        │                  Docker Network                  │
                        │                                                  │
 creditcard.csv ──► producer.py ──► [ Kafka : 9092 ]                      │
                                          │                                │
                              ┌───────────┴──────────────┐                │
                              ▼                          ▼                │
                     spark_streaming.py          spark_streaming.py       │
                     (feature engineering)       (raw transactions)       │
                              │                          │                │
                              ▼                          ▼                │
                     [ Redis : 6379 ]        hive_warehouse/ (Parquet)    │
                     features:<user_id>                                   │
                              │                                           │
                              └──────────────┐                            │
                                             ▼                            │
                              train_model.py ──► [ MLflow : 5001 ]        │
                              (Random Forest)     fraud_model v1          │
                                             │                            │
                                             ▼                            │
                              api.py  ──► /predict                        │
                              (FastAPI : 8000)                             │
                                             │                            │
                                             ▼                            │
                              predictions_log.csv                         │
                                             │                            │
                                             ▼                            │
                              drift_check.py ──► drift_report.png         │
                        └─────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data streaming | Apache Kafka + Zookeeper (Docker) |
| Feature engineering | PySpark Structured Streaming |
| Real-time feature store | Redis |
| Historical feature store | Apache Parquet (Hive-compatible) |
| Model tracking & registry | MLflow |
| ML model | scikit-learn Random Forest |
| Model serving API | FastAPI + Uvicorn |
| Drift monitoring | matplotlib + pandas |
| Language | Python 3.11 |

---

## Project Structure

```
PRODUCTION GRADE/
├── docker-compose.yml      # Kafka, Zookeeper, Redis, MLflow, Kafka-UI
├── producer.py             # Streams creditcard.csv → Kafka
├── spark_streaming.py      # Kafka → Redis features + Parquet
├── train_model.py          # Trains Random Forest → MLflow registry
├── api.py                  # FastAPI /predict endpoint
├── drift_check.py          # Fraud rate drift monitor + dashboard
├── requirements.txt        # All Python dependencies
├── creditcard.csv          # Kaggle Credit Card Fraud dataset (284k rows)
├── predictions_log.csv     # Auto-created by api.py at runtime
├── drift_report.png        # Auto-created by drift_check.py at runtime
├── mlflow_local/           # MLflow SQLite DB + model artifacts
│   ├── mlflow.db
│   └── artifacts/
├── hive_warehouse/         # Parquet files from Spark (date-partitioned)
│   └── transactions/
│       └── txn_date=YYYY-MM-DD/
└── checkpoints/            # Spark Structured Streaming checkpoints
    ├── redis/
    └── hive/
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | |
| Java (JDK) | 11 | Required by PySpark — [Download](https://adoptium.net) |
| Docker Desktop | Latest | For Kafka + Redis containers |
| creditcard.csv | — | See dataset section below |

### Dataset

Download from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and place `creditcard.csv` in the project root, **or** run the automated download:

```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p .
```

---

## Setup

### 1. Clone / navigate to project folder

```bash
cd "PRODUCTION GRADE"
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Set JAVA_HOME (required for PySpark)

```bash
# Windows
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-11.x.x
set PATH=%PATH%;%JAVA_HOME%\bin

# Mac / Linux
export JAVA_HOME=$(/usr/libexec/java_home -v 11)
```

### 4. Set HADOOP_HOME (Windows only — PySpark needs winutils.exe)

```bash
# Download winutils from https://github.com/cdarlint/winutils
# Then:
set HADOOP_HOME=C:\hadoop
set PATH=%PATH%;%HADOOP_HOME%\bin
```

### 5. Start Docker services

```bash
docker compose up -d
docker compose ps    # wait until all show "healthy"
```

Services started:

| Service | URL |
|---|---|
| Kafka | localhost:9092 |
| Kafka UI | http://localhost:8080 |
| Redis | localhost:6379 |
| MLflow UI | http://localhost:5001 |

---

## Running Each Script

### Terminal 1 — Train the model (do this first)

```bash
python train_model.py
```

Expected output:
```
[DATA] Sampled dataset : (50492, 31)  (fraud: 492)
[TRAIN] Done in ~8s
  precision  recall  f1-score
  Fraud       0.97    0.87    0.91
[MLFLOW] Version 1 promoted to 'Staging'.
```

### Terminal 2 — Start the prediction API

```bash
python api.py
# OR
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

API available at:
- Swagger UI → http://localhost:8000/docs
- Health check → http://localhost:8000/health
- Stats → http://localhost:8000/stats

### Terminal 3 — Stream transactions from the dataset

```bash
python producer.py
```

Sends one row every 0.5 seconds to the `transactions` Kafka topic.
Monitor live messages at http://localhost:8080.

### Terminal 4 — Compute features with PySpark Streaming

```bash
python spark_streaming.py
```

Reads from Kafka, computes 5-minute windowed features per user,
writes to Redis and Parquet. First run downloads the Kafka JAR (~30s).

### Terminal 5 — Make a fraud prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_0042",
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": 0.01,
    "Amount": 149.62
  }'
```

Response:
```json
{
  "user_id": "user_0042",
  "prediction": "LEGIT",
  "confidence": 0.976972,
  "fraud_prob": 0.023028,
  "legit_prob": 0.976972,
  "model_version": "1",
  "latency_ms": 57.96
}
```

### Run drift check at any time

```bash
python drift_check.py              # full report + drift_report.png
python drift_check.py --window 20  # custom rolling window
python drift_check.py --no-plot    # CI/CD mode (exits 1 on WARNING)
```

---

## Model Performance

Trained on a stratified sample of the Credit Card Fraud dataset (50k legit + 492 fraud):

| Metric | Value |
|---|---|
| Accuracy | 99.84% |
| Precision (Fraud) | 96.6% |
| Recall (Fraud) | 86.7% |
| F1 Score | 91.4% |
| ROC-AUC | 97.6% |

Class imbalance handled with `class_weight='balanced'`.
Top predictive features: `V14`, `V10`, `V12`, `V11`, `V17`.

---

## Drift Monitoring

`drift_check.py` compares live predictions against the dataset baseline (1.73% fraud).

| Status | Condition |
|---|---|
| OK | Live fraud rate ≤ 10% |
| CAUTION | Live rate drifted > 5% from baseline |
| WARNING | Live fraud rate > 10% — retrain recommended |

The drift report (`drift_report.png`) shows:
- Rolling fraud rate vs baseline + warning threshold
- Fraud probability score distribution
- Transaction amounts coloured by prediction outcome
- API prediction latency over time

---

## Stopping Everything

```bash
# Stop the API (Ctrl+C in its terminal)

# Stop Docker containers
docker compose down          # keeps MLflow data
docker compose down -v       # also deletes volumes
```
