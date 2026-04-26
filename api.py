"""
api.py
------
FastAPI model-serving endpoint for real-time fraud prediction.

Flow per request:
  1. Accept user_id + transaction features (V1-V28, Amount)
  2. Fetch live windowed features from Redis (txn_count, avg_amount, max_amount)
  3. Merge into a feature vector
  4. Run inference with the MLflow-registered Random Forest
  5. Return FRAUD / LEGIT + confidence score
  6. Append every prediction to predictions_log.csv for drift monitoring

Run:
    pip install fastapi uvicorn redis mlflow scikit-learn
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Test:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"user_id":"user_0042","V1":-1.36,"V2":-0.07,"V3":2.54,
              "V4":1.38,"V5":-0.34,"V6":0.46,"V7":0.24,"V8":0.10,
              "V9":0.36,"V10":0.09,"V11":-0.55,"V12":-0.62,"V13":-0.99,
              "V14":-0.31,"V15":1.47,"V16":-0.47,"V17":0.21,"V18":0.03,
              "V19":0.40,"V20":0.25,"V21":-0.02,"V22":0.28,"V23":-0.11,
              "V24":0.07,"V25":0.13,"V26":-0.19,"V27":0.13,"V28":0.01,
              "Amount":149.62}'
"""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import redis as redis_lib
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Configuration ─────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "sqlite:///C:/Users/ARYA/Desktop/PRODUCTION GRADE/mlflow_local/mlflow.db"
MODEL_NAME          = "fraud_model"
MODEL_STAGE         = "Staging"        # load the Staging-aliased version

REDIS_HOST          = "localhost"
REDIS_PORT          = 6379

LOG_FILE            = "predictions_log.csv"

# Feature columns the model was trained on (must match train_model.py exactly)
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


# ── Pydantic request / response schemas ───────────────────────────────────────
class TransactionRequest(BaseModel):
    user_id: str = Field(..., example="user_0042", description="Synthetic user identifier")
    # Core PCA features — all optional so partial requests still work
    V1:  float = 0.0; V2:  float = 0.0; V3:  float = 0.0; V4:  float = 0.0
    V5:  float = 0.0; V6:  float = 0.0; V7:  float = 0.0; V8:  float = 0.0
    V9:  float = 0.0; V10: float = 0.0; V11: float = 0.0; V12: float = 0.0
    V13: float = 0.0; V14: float = 0.0; V15: float = 0.0; V16: float = 0.0
    V17: float = 0.0; V18: float = 0.0; V19: float = 0.0; V20: float = 0.0
    V21: float = 0.0; V22: float = 0.0; V23: float = 0.0; V24: float = 0.0
    V25: float = 0.0; V26: float = 0.0; V27: float = 0.0; V28: float = 0.0
    Amount: float = Field(0.0, ge=0.0, description="Transaction amount in USD")


class PredictionResponse(BaseModel):
    user_id:        str
    prediction:     str            # "FRAUD" or "LEGIT"
    confidence:     float          # probability of FRAUD (0–1)
    fraud_prob:     float
    legit_prob:     float
    redis_features: dict           # windowed features pulled from Redis (or defaults)
    model_version:  str
    latency_ms:     float


# ── Application setup ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud prediction powered by MLflow + Redis feature store",
    version="1.0.0",
)

# Global holders — populated once at startup, reused across requests
_model       = None
_model_meta  = {"version": "unknown", "run_id": "unknown"}
_redis_client: Optional[redis_lib.Redis] = None


# ── Startup: load model + connect Redis ───────────────────────────────────────
@app.on_event("startup")
def startup():
    """Load the fraud model from MLflow registry and connect to Redis."""
    global _model, _model_meta, _redis_client

    # ── Load model from MLflow registry ───────────────────────────────────
    print(f"[STARTUP] Loading model '{MODEL_NAME}' (stage={MODEL_STAGE}) ...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        # Try stage-based URI first
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        _model    = mlflow.sklearn.load_model(model_uri)
        print(f"[STARTUP] Model loaded from: {model_uri}")
    except Exception:
        try:
            # Fallback: load latest version regardless of stage
            model_uri = f"models:/{MODEL_NAME}/1"
            _model    = mlflow.sklearn.load_model(model_uri)
            print(f"[STARTUP] Model loaded from fallback URI: {model_uri}")
        except Exception as e:
            print(f"[STARTUP ERROR] Could not load model: {e}")
            print("[STARTUP] API will start but /predict will fail until model is available.")
            _model = None

    # Capture model metadata for response enrichment
    try:
        client  = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        version = client.get_registered_model(MODEL_NAME).latest_versions
        if version:
            _model_meta["version"] = version[0].version
            _model_meta["run_id"]  = version[0].run_id
    except Exception:
        pass

    # ── Connect to Redis ───────────────────────────────────────────────────
    try:
        _redis_client = redis_lib.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True, socket_timeout=1
        )
        _redis_client.ping()
        print(f"[STARTUP] Redis connected at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        print(f"[STARTUP] Redis not available ({e}) — will use default features.")
        _redis_client = None

    # ── Ensure predictions log file has a header ───────────────────────────
    if not Path(LOG_FILE).exists():
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "user_id", "prediction", "confidence",
                "fraud_prob", "legit_prob", "amount",
                "txn_count", "avg_amount", "max_amount",
                "model_version", "latency_ms",
            ])
        print(f"[STARTUP] Created predictions log: {LOG_FILE}")

    print("[STARTUP] API ready.")


# ── Redis feature fetch ────────────────────────────────────────────────────────
def fetch_redis_features(user_id: str) -> dict:
    """
    Pull the latest windowed features for user_id from Redis.
    Returns sensible defaults if Redis is down or the key doesn't exist yet
    (e.g. a brand-new user the streaming job hasn't seen yet).
    """
    defaults = {"txn_count": 1, "avg_amount": 0.0, "max_amount": 0.0}

    if _redis_client is None:
        return {**defaults, "source": "default (redis unavailable)"}

    try:
        raw = _redis_client.get(f"features:{user_id}")
        if raw is None:
            return {**defaults, "source": "default (key not found)"}
        data = json.loads(raw)
        return {
            "txn_count":  data.get("txn_count",  defaults["txn_count"]),
            "avg_amount": data.get("avg_amount",  defaults["avg_amount"]),
            "max_amount": data.get("max_amount",  defaults["max_amount"]),
            "window_start": data.get("window_start", ""),
            "window_end":   data.get("window_end",   ""),
            "source": "redis",
        }
    except Exception as e:
        return {**defaults, "source": f"default (error: {e})"}


# ── Feature vector builder ─────────────────────────────────────────────────────
def build_feature_vector(req: TransactionRequest) -> np.ndarray:
    """
    Construct the 29-element feature vector [V1..V28, Amount] in the exact
    order the Random Forest was trained on.
    """
    return np.array(
        [getattr(req, col) for col in FEATURE_COLS],
        dtype=np.float32,
    ).reshape(1, -1)


# ── Prediction logger ─────────────────────────────────────────────────────────
def log_prediction(
    user_id: str,
    prediction: str,
    confidence: float,
    fraud_prob: float,
    legit_prob: float,
    amount: float,
    redis_features: dict,
    model_version: str,
    latency_ms: float,
):
    """Append one row to predictions_log.csv for downstream drift analysis."""
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            user_id,
            prediction,
            round(confidence, 6),
            round(fraud_prob, 6),
            round(legit_prob, 6),
            round(amount, 4),
            redis_features.get("txn_count",  0),
            redis_features.get("avg_amount", 0.0),
            redis_features.get("max_amount", 0.0),
            model_version,
            round(latency_ms, 2),
        ])


# ── Main prediction endpoint ──────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(req: TransactionRequest):
    """
    Predict whether a transaction is FRAUD or LEGIT.

    - Fetches live per-user features from Redis (txn velocity, avg/max amount)
    - Combines with the raw V1-V28 + Amount features from the request
    - Runs Random Forest inference
    - Logs the result to predictions_log.csv
    """
    t_start = time.perf_counter()

    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train_model.py first.",
        )

    # ── 1. Fetch Redis features for this user ──────────────────────────────
    redis_features = fetch_redis_features(req.user_id)

    # ── 2. Build feature vector ────────────────────────────────────────────
    X = build_feature_vector(req)

    # ── 3. Run inference ───────────────────────────────────────────────────
    try:
        proba      = _model.predict_proba(X)[0]   # [P(legit), P(fraud)]
        legit_prob = float(proba[0])
        fraud_prob = float(proba[1])
        prediction = "FRAUD" if fraud_prob >= 0.5 else "LEGIT"
        confidence = fraud_prob if prediction == "FRAUD" else legit_prob
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    latency_ms = (time.perf_counter() - t_start) * 1000

    # ── 4. Log to CSV ──────────────────────────────────────────────────────
    log_prediction(
        user_id        = req.user_id,
        prediction     = prediction,
        confidence     = confidence,
        fraud_prob     = fraud_prob,
        legit_prob     = legit_prob,
        amount         = req.Amount,
        redis_features = redis_features,
        model_version  = _model_meta["version"],
        latency_ms     = latency_ms,
    )

    # ── 5. Print to terminal ───────────────────────────────────────────────
    flag = "⚠  FRAUD" if prediction == "FRAUD" else "   legit"
    print(
        f"[PREDICT] {flag}  user={req.user_id}  "
        f"amount=${req.Amount:.2f}  "
        f"fraud_prob={fraud_prob:.4f}  "
        f"latency={latency_ms:.1f}ms  "
        f"redis={redis_features.get('source','?')}"
    )

    return PredictionResponse(
        user_id        = req.user_id,
        prediction     = prediction,
        confidence     = round(confidence, 6),
        fraud_prob     = round(fraud_prob, 6),
        legit_prob     = round(legit_prob, 6),
        redis_features = redis_features,
        model_version  = str(_model_meta["version"]),
        latency_ms     = round(latency_ms, 2),
    )


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Returns service status and whether the model + Redis are available."""
    redis_ok = False
    if _redis_client:
        try:
            _redis_client.ping()
            redis_ok = True
        except Exception:
            pass

    return {
        "status":        "ok",
        "model_loaded":  _model is not None,
        "model_version": _model_meta["version"],
        "redis_ok":      redis_ok,
        "log_file":      LOG_FILE,
    }


# ── Predictions summary endpoint ──────────────────────────────────────────────
@app.get("/stats")
def stats():
    """Quick summary of predictions logged so far."""
    if not Path(LOG_FILE).exists():
        return {"total": 0, "fraud": 0, "legit": 0, "fraud_rate_pct": 0.0}

    total = fraud = 0
    with open(LOG_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row.get("prediction") == "FRAUD":
                fraud += 1

    legit       = total - fraud
    fraud_rate  = round((fraud / total * 100) if total > 0 else 0.0, 3)
    return {
        "total":          total,
        "fraud":          fraud,
        "legit":          legit,
        "fraud_rate_pct": fraud_rate,
        "baseline_pct":   1.73,
        "log_file":       LOG_FILE,
    }


# ── Entry point (for running directly with `python api.py`) ───────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
