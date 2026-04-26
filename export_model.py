"""
export_model.py
---------------
One-time script: loads the fraud_model from MLflow and saves it
as model.joblib so the Streamlit demo can run without an MLflow server.
"""
import joblib
import mlflow.sklearn

MLFLOW_TRACKING_URI = "sqlite:///C:/Users/ARYA/Desktop/PRODUCTION GRADE/mlflow_local/mlflow.db"
OUTPUT_PATH = "C:/Users/ARYA/Desktop/PRODUCTION GRADE/model.joblib"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("[EXPORT] Loading model from MLflow registry ...")
try:
    model = mlflow.sklearn.load_model("models:/fraud_model/Staging")
except Exception:
    model = mlflow.sklearn.load_model("models:/fraud_model/1")

joblib.dump(model, OUTPUT_PATH, compress=3)
print(f"[EXPORT] Saved to: {OUTPUT_PATH}")

import os
size_mb = os.path.getsize(OUTPUT_PATH) / 1_048_576
print(f"[EXPORT] File size: {size_mb:.1f} MB")
