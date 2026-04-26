"""
train_model.py
--------------
Trains a Random Forest fraud classifier on creditcard.csv, logs all
metrics + artifacts to MLflow, and registers the final model in the
MLflow Model Registry as "fraud_model".

Run AFTER: docker compose up -d  (MLflow server must be reachable)

Usage:
    python train_model.py
"""

import sys
import time
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ── Configuration ─────────────────────────────────────────────────────────────
# Use local file-based tracking so artifacts write directly to disk on Windows.
# The MLflow UI server reads this same SQLite DB, so runs appear there too.
MLFLOW_TRACKING_URI = "sqlite:///C:/Users/ARYA/Desktop/PRODUCTION GRADE/mlflow_local/mlflow.db"
MLFLOW_ARTIFACT_ROOT = "C:/Users/ARYA/Desktop/PRODUCTION GRADE/mlflow_local/artifacts"
EXPERIMENT_NAME     = "fraud-detection"
MODEL_NAME          = "fraud_model"             # registry name
CSV_PATH            = "creditcard.csv"
TEST_SIZE           = 0.2
RANDOM_STATE        = 42

# Random Forest hyperparameters
RF_PARAMS = {
    "n_estimators":  50,           # reduced from 100 to save RAM
    "max_depth":     10,           # shallower trees use less memory
    "min_samples_split": 5,
    "min_samples_leaf":  2,
    "class_weight":  "balanced",   # compensates for ~0.17% fraud class
    "n_jobs":        2,            # limit parallelism to reduce peak RAM
    "random_state":  RANDOM_STATE,
}

# Features used for training (V1-V28 + Amount; Time excluded — not predictive)
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COL   = "Class"


# ── Data loading & preparation ────────────────────────────────────────────────
def load_and_prepare(csv_path: str):
    """
    Load creditcard.csv, scale Amount, split into train/test.
    Time is dropped — it's a sequence counter, not a real feature.
    Amount is scaled because it has a very different magnitude than V1-V28.
    """
    print(f"[DATA] Loading '{csv_path}' ...")
    # Read in chunks with float32 to stay within memory limits
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=50_000, dtype=np.float32):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    print(f"[DATA] Full dataset shape: {df.shape}")
    fraud_pct = df[TARGET_COL].mean() * 100
    print(f"[DATA] Fraud rate: {fraud_pct:.3f}%")

    # Stratified sample to reduce peak RAM — keep all 492 fraud rows
    # and sample 50k legit rows. Model quality is nearly identical to full data.
    fraud_df  = df[df[TARGET_COL] == 1.0]
    legit_df  = df[df[TARGET_COL] == 0.0].sample(n=50_000, random_state=RANDOM_STATE)
    df        = pd.concat([fraud_df, legit_df], ignore_index=True).sample(
                    frac=1, random_state=RANDOM_STATE)  # shuffle
    print(f"[DATA] Sampled dataset : {df.shape}  (fraud: {int(df[TARGET_COL].sum())})\n")

    # Scale Amount to bring it in line with PCA features V1-V28
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,   # preserve fraud/legit ratio in both splits
    )

    print(f"[DATA] Train size : {len(X_train):,}  (fraud: {y_train.sum()})")
    print(f"[DATA] Test  size : {len(X_test):,}  (fraud: {y_test.sum()})\n")

    return X_train, X_test, y_train, y_test, scaler


# ── Model training ────────────────────────────────────────────────────────────
def train(X_train, y_train) -> RandomForestClassifier:
    """Fit a Random Forest with class_weight='balanced' to handle skew."""
    print("[TRAIN] Fitting Random Forest ...")
    print(f"[TRAIN] Params: {RF_PARAMS}\n")
    t0 = time.time()
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[TRAIN] Done in {elapsed:.1f}s\n")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test) -> dict:
    """
    Compute metrics on the held-out test set.
    Returns a dict ready to be logged to MLflow.
    """
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 6),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 6),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 6),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 6),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 6),
    }

    print("=" * 55)
    print("  CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    print("CONFUSION MATRIX")
    print("-" * 30)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  True  Negatives (correct legit)  : {tn:>6,}")
    print(f"  False Positives (false alarm)     : {fp:>6,}")
    print(f"  False Negatives (missed fraud)    : {fn:>6,}")
    print(f"  True  Positives (caught fraud)    : {tp:>6,}")
    print()

    print("METRICS")
    print("-" * 30)
    for k, v in metrics.items():
        print(f"  {k:<12}: {v}")
    print()

    return metrics, y_pred, y_prob


# ── Feature importance ────────────────────────────────────────────────────────
def top_features(model, n: int = 10) -> pd.DataFrame:
    """Return the top-N most important features as a DataFrame."""
    importance = pd.DataFrame({
        "feature":   FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(n)
    print(f"TOP {n} FEATURES")
    print("-" * 30)
    print(importance.to_string(index=False))
    print()
    return importance


# ── MLflow logging ────────────────────────────────────────────────────────────
def log_to_mlflow(model, metrics: dict, X_train, y_train, importance_df):
    """
    Log everything to MLflow and register the model in the Model Registry.

    Run steps:
      1. Log hyperparameters
      2. Log evaluation metrics
      3. Log feature importance as a CSV artifact
      4. Log the trained model with input/output signature
      5. Register model as "fraud_model" and transition to Staging
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Create experiment with an explicit file:/// artifact location so MLflow
    # can resolve it on Windows without a running HTTP server.
    artifact_uri = "file:///" + MLFLOW_ARTIFACT_ROOT.replace("\\", "/").lstrip("/")
    try:
        exp_id = mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_uri)
    except mlflow.exceptions.MlflowException:
        exp_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    mlflow.set_experiment(experiment_id=exp_id)

    with mlflow.start_run(run_name="random_forest_v1") as run:
        run_id = run.info.run_id
        print(f"[MLFLOW] Run ID: {run_id}")

        # ── 1. Log hyperparameters ─────────────────────────────────────────
        mlflow.log_params(RF_PARAMS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("features",  ",".join(FEATURE_COLS))
        mlflow.log_param("n_features", len(FEATURE_COLS))

        # ── 2. Log metrics ─────────────────────────────────────────────────
        mlflow.log_metrics(metrics)
        print(f"[MLFLOW] Logged params + metrics.")

        # ── 3. Log feature importance as artifact ──────────────────────────
        importance_path = "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        print(f"[MLFLOW] Logged feature importance artifact.")

        # ── 4. Log the model with signature ───────────────────────────────
        # Signature captures input column types + output shape so the
        # serving layer can validate requests automatically.
        input_example = pd.DataFrame(X_train[:3], columns=FEATURE_COLS)
        signature     = infer_signature(
            input_example,
            model.predict(X_train[:3])
        )
        mlflow.sklearn.log_model(
            sk_model       = model,
            artifact_path  = "model",
            signature      = signature,
            input_example  = input_example,
            registered_model_name = MODEL_NAME,   # auto-creates registry entry
        )
        print(f"[MLFLOW] Model logged and registered as '{MODEL_NAME}'.")

        model_uri = f"runs:/{run_id}/model"

    return run_id, model_uri


# ── Transition model to Staging ───────────────────────────────────────────────
def promote_to_staging(model_name: str):
    """
    Find the latest registered version and move it to 'Staging'.
    In production you would add validation gates before promoting to Production.
    """
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Get all versions of the model
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print("[MLFLOW] No model versions found — skipping staging promotion.")
        return

    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    print(f"[MLFLOW] Latest version: {latest.version}  status: {latest.current_stage}")

    try:
        client.transition_model_version_stage(
            name    = model_name,
            version = latest.version,
            stage   = "Staging",
            archive_existing_versions = True,   # archive older Staging versions
        )
        print(f"[MLFLOW] Version {latest.version} promoted to 'Staging'.")
    except Exception as e:
        # Newer MLflow versions deprecate stages in favour of aliases
        try:
            client.set_registered_model_alias(model_name, "staging", latest.version)
            print(f"[MLFLOW] Set alias 'staging' → version {latest.version}.")
        except Exception as e2:
            print(f"[MLFLOW] Could not promote (non-fatal): {e2}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Fraud Detection — Model Training")
    print("=" * 55)
    print()

    # Verify MLflow DB directory exists (file-based tracking needs no server)
    import os
    os.makedirs(MLFLOW_ARTIFACT_ROOT, exist_ok=True)
    print(f"[OK] MLflow tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"[OK] Artifact root       : {MLFLOW_ARTIFACT_ROOT}\n")

    # Pipeline
    X_train, X_test, y_train, y_test, scaler = load_and_prepare(CSV_PATH)
    model                                    = train(X_train, y_train)
    metrics, y_pred, y_prob                  = evaluate(model, X_test, y_test)
    importance_df                            = top_features(model)
    run_id, model_uri                        = log_to_mlflow(
                                                   model, metrics,
                                                   X_train, y_train,
                                                   importance_df
                                               )
    promote_to_staging(MODEL_NAME)

    print()
    print("=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55)
    print(f"  MLflow Run ID : {run_id}")
    print(f"  Model URI     : {model_uri}")
    print(f"  Registry name : {MODEL_NAME}")
    print(f"  View UI       : {MLFLOW_TRACKING_URI}")
    print("=" * 55)


if __name__ == "__main__":
    main()
