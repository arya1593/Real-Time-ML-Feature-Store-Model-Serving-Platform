"""
drift_check.py
--------------
Reads predictions_log.csv and checks for model/data drift by comparing
the live fraud prediction rate against the dataset baseline (~1.73%).

What it checks:
  1. Overall fraud rate vs baseline
  2. Rolling fraud rate over time (10-prediction sliding window)
  3. Confidence score distribution (low confidence = model uncertainty)
  4. Flags WARNING if current fraud rate exceeds 10%

Also produces a 4-panel matplotlib dashboard saved to drift_report.png.

Run:
    python drift_check.py
    python drift_check.py --window 20   (custom rolling window)
"""

import argparse
import csv
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
LOG_FILE            = "predictions_log.csv"
BASELINE_FRAUD_RATE = 1.73      # % fraud in the original creditcard.csv dataset
WARNING_THRESHOLD   = 10.0      # % — alert if live rate exceeds this
REPORT_IMAGE        = "drift_report.png"


# ── Data loading ──────────────────────────────────────────────────────────────
def load_predictions(path: str) -> list[dict]:
    """
    Load predictions_log.csv into a list of dicts.
    Skips malformed rows silently.
    """
    if not Path(path).exists():
        print(f"[ERROR] Log file not found: {path}")
        print("  → Run api.py and make some /predict calls first.")
        sys.exit(1)

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                rows.append({
                    "timestamp":   datetime.fromisoformat(row["timestamp"]),
                    "user_id":     row["user_id"],
                    "prediction":  row["prediction"],
                    "confidence":  float(row["confidence"]),
                    "fraud_prob":  float(row["fraud_prob"]),
                    "legit_prob":  float(row["legit_prob"]),
                    "amount":      float(row["amount"]),
                    "txn_count":   int(float(row.get("txn_count", 0))),
                    "avg_amount":  float(row.get("avg_amount", 0)),
                    "max_amount":  float(row.get("max_amount", 0)),
                    "model_version": row.get("model_version", "?"),
                    "latency_ms":  float(row.get("latency_ms", 0)),
                    "is_fraud":    row["prediction"] == "FRAUD",
                })
            except (ValueError, KeyError):
                pass   # skip bad rows

    return rows


# ── Summary statistics ────────────────────────────────────────────────────────
def compute_summary(rows: list[dict]) -> dict:
    total      = len(rows)
    fraud      = sum(1 for r in rows if r["is_fraud"])
    legit      = total - fraud
    fraud_rate = (fraud / total * 100) if total > 0 else 0.0

    confidences  = [r["confidence"] for r in rows]
    fraud_probs  = [r["fraud_prob"] for r in rows]
    amounts      = [r["amount"]     for r in rows]
    latencies    = [r["latency_ms"] for r in rows]

    return {
        "total":            total,
        "fraud":            fraud,
        "legit":            legit,
        "fraud_rate_pct":   round(fraud_rate, 4),
        "avg_confidence":   round(np.mean(confidences), 4)  if confidences else 0,
        "min_confidence":   round(np.min(confidences),  4)  if confidences else 0,
        "avg_fraud_prob":   round(np.mean(fraud_probs), 4)  if fraud_probs else 0,
        "avg_amount":       round(np.mean(amounts),     2)  if amounts     else 0,
        "avg_latency_ms":   round(np.mean(latencies),   2)  if latencies   else 0,
        "model_version":    rows[-1]["model_version"]        if rows        else "?",
        "first_seen":       rows[0]["timestamp"].isoformat() if rows        else "—",
        "last_seen":        rows[-1]["timestamp"].isoformat() if rows       else "—",
    }


# ── Rolling fraud rate ────────────────────────────────────────────────────────
def rolling_fraud_rate(rows: list[dict], window: int) -> tuple[list, list]:
    """
    Compute a sliding-window fraud rate over the sorted predictions.
    Returns (timestamps, fraud_rate_pct_list).
    """
    buf        = deque(maxlen=window)
    timestamps = []
    rates      = []

    for row in rows:
        buf.append(int(row["is_fraud"]))
        if len(buf) == window:
            rate = sum(buf) / window * 100
            timestamps.append(row["timestamp"])
            rates.append(rate)

    return timestamps, rates


# ── Drift report printer ──────────────────────────────────────────────────────
def print_report(summary: dict, rows: list[dict], window: int):
    fraud_rate = summary["fraud_rate_pct"]
    delta      = fraud_rate - BASELINE_FRAUD_RATE
    status     = "OK" if fraud_rate <= WARNING_THRESHOLD else "WARNING"

    print()
    print("=" * 60)
    print("  DRIFT CHECK REPORT")
    print("=" * 60)
    print(f"  Log file         : {LOG_FILE}")
    print(f"  Period           : {summary['first_seen']}  →  {summary['last_seen']}")
    print(f"  Model version    : {summary['model_version']}")
    print()
    print(f"  Total predictions: {summary['total']:,}")
    print(f"  Fraud            : {summary['fraud']:,}  ({fraud_rate:.3f}%)")
    print(f"  Legit            : {summary['legit']:,}")
    print()
    print(f"  Baseline fraud rate  : {BASELINE_FRAUD_RATE:.2f}%")
    print(f"  Live    fraud rate   : {fraud_rate:.3f}%")
    drift_symbol = "+" if delta >= 0 else ""
    print(f"  Delta vs baseline    : {drift_symbol}{delta:.3f}%")
    print()
    print(f"  Avg confidence   : {summary['avg_confidence']:.4f}")
    print(f"  Min confidence   : {summary['min_confidence']:.4f}  "
          f"({'OK' if summary['min_confidence'] > 0.6 else 'LOW — model uncertain'})")
    print(f"  Avg fraud prob   : {summary['avg_fraud_prob']:.4f}")
    print(f"  Avg amount       : ${summary['avg_amount']:.2f}")
    print(f"  Avg latency      : {summary['avg_latency_ms']:.1f}ms")
    print()

    # ── Drift verdict ──────────────────────────────────────────────────────
    if fraud_rate > WARNING_THRESHOLD:
        print(f"  *** WARNING: Fraud rate {fraud_rate:.2f}% exceeds threshold "
              f"{WARNING_THRESHOLD:.1f}% ***")
        print("  ACTION: Retrain model or investigate incoming data distribution.")
    elif abs(delta) > 5:
        print(f"  [CAUTION] Fraud rate drifted {abs(delta):.2f}% from baseline.")
        print("  ACTION: Monitor closely. Consider retraining if trend continues.")
    else:
        print("  [OK] Fraud rate within acceptable range of baseline.")

    print()

    # ── Rolling window summary ─────────────────────────────────────────────
    if summary["total"] >= window:
        ts, rates = rolling_fraud_rate(rows, window)
        if rates:
            print(f"  Rolling {window}-prediction fraud rate:")
            print(f"    current  : {rates[-1]:.2f}%")
            print(f"    min      : {min(rates):.2f}%")
            print(f"    max      : {max(rates):.2f}%")
            peak_idx = rates.index(max(rates))
            print(f"    peak at  : {ts[peak_idx].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"  [NOTE] Need {window} predictions for rolling window "
              f"(have {summary['total']}).")

    print("=" * 60)
    print()

    return status


# ── Matplotlib dashboard ──────────────────────────────────────────────────────
def plot_dashboard(rows: list[dict], summary: dict, window: int, out_path: str):
    """
    4-panel drift monitoring dashboard:
      [0,0] Rolling fraud rate vs baseline & warning threshold
      [0,1] Fraud probability distribution (histogram)
      [1,0] Transaction amount over time coloured by prediction
      [1,1] Prediction latency over time
    """
    if len(rows) < 2:
        print("[PLOT] Need at least 2 predictions to plot. Skipping.")
        return

    timestamps = [r["timestamp"] for r in rows]
    is_fraud   = [r["is_fraud"]  for r in rows]
    amounts    = [r["amount"]    for r in rows]
    fraud_prob = [r["fraud_prob"] for r in rows]
    latencies  = [r["latency_ms"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"Fraud Detection — Drift Report  |  "
        f"Live rate: {summary['fraud_rate_pct']:.2f}%  |  "
        f"Baseline: {BASELINE_FRAUD_RATE:.2f}%",
        fontsize=13, fontweight="bold"
    )

    # ── Panel 0,0: Rolling fraud rate ─────────────────────────────────────
    ax = axes[0, 0]
    win = min(window, len(rows))
    ts_roll, rate_roll = rolling_fraud_rate(rows, win)

    if ts_roll:
        ax.plot(ts_roll, rate_roll, color="steelblue", linewidth=2,
                label=f"Rolling {win}-pred rate")
    ax.axhline(BASELINE_FRAUD_RATE, color="green",  linestyle="--",
               linewidth=1.5, label=f"Baseline {BASELINE_FRAUD_RATE}%")
    ax.axhline(WARNING_THRESHOLD,   color="red",    linestyle="--",
               linewidth=1.5, label=f"Warning {WARNING_THRESHOLD}%")

    ax.set_title("Rolling Fraud Rate")
    ax.set_xlabel("Time")
    ax.set_ylabel("Fraud rate (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Shade the warning zone
    ax.axhspan(WARNING_THRESHOLD, ax.get_ylim()[1] if rate_roll else WARNING_THRESHOLD + 5,
               alpha=0.08, color="red")

    # ── Panel 0,1: Fraud probability histogram ─────────────────────────────
    ax = axes[0, 1]
    fraud_vals = [p for p, f in zip(fraud_prob, is_fraud) if f]
    legit_vals = [p for p, f in zip(fraud_prob, is_fraud) if not f]

    if legit_vals:
        ax.hist(legit_vals, bins=20, alpha=0.6, color="steelblue",
                label="Legit predictions", density=True)
    if fraud_vals:
        ax.hist(fraud_vals, bins=20, alpha=0.6, color="red",
                label="Fraud predictions", density=True)

    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Decision threshold")
    ax.set_title("Fraud Probability Distribution")
    ax.set_xlabel("P(fraud)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel 1,0: Amount over time coloured by prediction ────────────────
    ax = axes[1, 0]
    legit_ts  = [t for t, f in zip(timestamps, is_fraud) if not f]
    legit_amt = [a for a, f in zip(amounts,    is_fraud) if not f]
    fraud_ts  = [t for t, f in zip(timestamps, is_fraud) if f]
    fraud_amt = [a for a, f in zip(amounts,    is_fraud) if f]

    if legit_ts:
        ax.scatter(legit_ts, legit_amt, c="steelblue", s=30, alpha=0.7,
                   label="Legit", zorder=2)
    if fraud_ts:
        ax.scatter(fraud_ts, fraud_amt, c="red", s=60, marker="x",
                   alpha=0.9, label="FRAUD", zorder=3)

    ax.set_title("Transaction Amount Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amount ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel 1,1: Prediction latency over time ────────────────────────────
    ax = axes[1, 1]
    ax.plot(timestamps, latencies, color="purple", linewidth=1.2, alpha=0.8)
    ax.fill_between(timestamps, latencies, alpha=0.15, color="purple")
    avg_lat = np.mean(latencies)
    ax.axhline(avg_lat, color="orange", linestyle="--",
               linewidth=1, label=f"Avg {avg_lat:.1f}ms")

    ax.set_title("Prediction Latency")
    ax.set_xlabel("Time")
    ax.set_ylabel("Latency (ms)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Dashboard saved to: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Model drift checker")
    parser.add_argument("--log",    default=LOG_FILE,  help="Path to predictions_log.csv")
    parser.add_argument("--window", default=10, type=int, help="Rolling window size")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    print(f"\n[DRIFT] Reading predictions from: {args.log}")
    rows    = load_predictions(args.log)
    summary = compute_summary(rows)
    status  = print_report(summary, rows, window=args.window)

    if not args.no_plot:
        try:
            plot_dashboard(rows, summary, window=args.window, out_path=REPORT_IMAGE)
        except Exception as e:
            print(f"[PLOT ERROR] Could not generate plot: {e}")

    # Exit with code 1 if drift is detected — useful for CI/CD gates
    sys.exit(1 if status == "WARNING" else 0)


if __name__ == "__main__":
    main()
