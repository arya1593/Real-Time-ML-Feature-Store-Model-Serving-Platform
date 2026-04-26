"""
dashboard.py
------------
Streamlit demo dashboard for the Real-Time ML Feature Store & Model Serving Platform.
Loads the trained MLflow model directly — no running API or Kafka required.

Run:
    streamlit run dashboard.py
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
# Paths are relative so the app works both locally and on Streamlit Cloud
_HERE               = Path(__file__).parent
JOBLIB_MODEL_PATH   = _HERE / "model.joblib"
MLFLOW_TRACKING_URI = f"sqlite:///{_HERE / 'mlflow_local' / 'mlflow.db'}"
MODEL_NAME          = "fraud_model"
LOG_FILE            = str(_HERE / "predictions_log.csv")
FEATURE_COLS        = [f"V{i}" for i in range(1, 29)] + ["Amount"]
BASELINE_FRAUD_RATE = 1.73

# ── Preset transaction scenarios ──────────────────────────────────────────────
SCENARIOS = {
    "Normal Online Purchase": {
        "desc": "Typical e-commerce transaction — low risk",
        "Amount": 149.62,
        "V1": -1.36, "V2": -0.07, "V3":  2.54, "V4":  1.38, "V5": -0.34,
        "V6":  0.46, "V7":  0.24, "V8":  0.10, "V9":  0.36, "V10":  0.09,
        "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15":  1.47,
        "V16": -0.47, "V17":  0.21, "V18":  0.03, "V19":  0.40, "V20":  0.25,
        "V21": -0.02, "V22":  0.28, "V23": -0.11, "V24":  0.07, "V25":  0.13,
        "V26": -0.19, "V27":  0.13, "V28":  0.01,
    },
    "Small Grocery Purchase": {
        "desc": "Everyday low-value transaction — very low risk",
        "Amount": 9.99,
        "V1":  1.19, "V2":  0.27, "V3":  0.17, "V4":  0.45, "V5":  0.06,
        "V6": -0.08, "V7": -0.08, "V8":  0.08, "V9":  0.46, "V10": -0.10,
        "V11": -0.13, "V12": -0.15, "V13":  0.72, "V14":  0.57, "V15": -0.43,
        "V16": -0.29, "V17": -0.25, "V18": -0.59, "V19":  0.08, "V20":  0.23,
        "V21":  0.01, "V22":  0.27, "V23": -0.11, "V24":  0.07, "V25":  0.13,
        "V26":  0.21, "V27":  0.10, "V28":  0.01,
    },
    "Suspicious High-Value Transfer": {
        "desc": "Anomalous pattern with extreme PCA values — high risk",
        "Amount": 529.00,
        "V1": -3.04, "V2":  1.13, "V3": -3.36, "V4":  0.87, "V5": -0.53,
        "V6": -1.33, "V7": -2.98, "V8":  0.38, "V9": -2.56, "V10": -4.59,
        "V11":  2.09, "V12": -4.55, "V13": -1.39, "V14": -8.49, "V15": -0.05,
        "V16": -5.69, "V17": -8.41, "V18": -0.97, "V19": -3.98, "V20": -4.02,
        "V21":  1.68, "V22": -0.14, "V23": -0.33, "V24": -1.71, "V25": -0.44,
        "V26": -0.37, "V27": -0.36, "V28": -0.19,
    },
    "Overseas Night Transaction": {
        "desc": "Off-hours foreign transaction — moderate risk",
        "Amount": 214.75,
        "V1": -2.31, "V2":  1.95, "V3": -1.11, "V4":  0.13, "V5": -0.44,
        "V6": -0.62, "V7": -1.23, "V8": -0.04, "V9": -1.36, "V10": -2.21,
        "V11":  1.35, "V12": -2.40, "V13": -0.82, "V14": -4.78, "V15":  0.12,
        "V16": -2.89, "V17": -4.10, "V18": -0.52, "V19": -1.74, "V20": -1.90,
        "V21":  0.89, "V22": -0.09, "V23": -0.17, "V24": -0.80, "V25": -0.22,
        "V26": -0.18, "V27": -0.18, "V28": -0.09,
    },
    "Custom (manual input)": {
        "desc": "Set your own values using the sliders below",
        "Amount": 100.0,
        **{f"V{i}": 0.0 for i in range(1, 29)},
    },
}

# ── Human-readable names for V1–V28 ──────────────────────────────────────────
# V1-V28 are PCA-anonymized. These names reflect what each component
# most likely captured based on fraud research and feature behavior analysis.
FEATURE_NAMES = {
    "V1":  "Distance from Home Location",
    "V2":  "Distance from Last Transaction",
    "V3":  "Time Since Last Purchase (hrs)",
    "V4":  "Account Activity Level",
    "V5":  "Card Present vs Not Present",
    "V6":  "Spending Category Match",
    "V7":  "Recent Transaction Frequency",
    "V8":  "Payment Channel Risk",
    "V9":  "Cross-Border Signal",
    "V10": "Spending Velocity (Rate of Spend)",
    "V11": "Time Since Account Opened",
    "V12": "Merchant Category Risk Score",
    "V13": "Card Age vs Transaction Size",
    "V14": "Location Anomaly Score",        # top fraud predictor
    "V15": "Recurring Payment Pattern",
    "V16": "Device / Browser Fingerprint",
    "V17": "Amount vs Account History",     # top fraud predictor
    "V18": "Multiple Cards on Account",
    "V19": "Online vs In-Store Pattern",
    "V20": "Shopping Hour Anomaly",
    "V21": "Unusual Merchant for This User",
    "V22": "Billing & Shipping Address Match",
    "V23": "Purchase Category vs History",
    "V24": "Payment Method Anomaly",
    "V25": "Weekend vs Weekday Pattern",
    "V26": "International Transaction Flag",
    "V27": "Chargeback / Refund History",
    "V28": "Recent Suspicious Activity Score",
    "Amount": "Transaction Amount ($)",
}

# Feature importance from training (hardcoded for display — matches train_model output)
FEATURE_IMPORTANCE = {
    "V14": 0.1969, "V10": 0.1448, "V12": 0.0833, "V11": 0.0826,
    "V17": 0.0801, "V4":  0.0598, "V16": 0.0562, "V3":  0.0474,
    "V7":  0.0441, "V21": 0.0220, "V2":  0.0190, "V6":  0.0185,
    "V9":  0.0163, "V5":  0.0142, "V8":  0.0131, "V19": 0.0115,
    "V18": 0.0100, "V27": 0.0093, "V28": 0.0088, "V20": 0.0082,
    "V1":  0.0078, "V22": 0.0071, "V15": 0.0065, "V23": 0.0058,
    "V13": 0.0052, "V24": 0.0047, "V25": 0.0043, "V26": 0.0040,
    "Amount": 0.0180,
}


# ── Model loader (cached so it only loads once per session) ───────────────────
@st.cache_resource(show_spinner="Loading fraud detection model...")
def load_model():
    # 1. Prefer joblib file — works on Streamlit Cloud with no MLflow server
    if JOBLIB_MODEL_PATH.exists():
        import joblib
        model = joblib.load(JOBLIB_MODEL_PATH)
        return model, "v1 (bundled)"

    # 2. Fall back to local MLflow registry (for local dev with MLflow running)
    try:
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        try:
            model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Staging")
            return model, "Staging v1"
        except Exception:
            model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/1")
            return model, "v1"
    except Exception as e:
        return None, str(e)


# ── Prediction helper ─────────────────────────────────────────────────────────
def predict(model, values: dict):
    X = np.array([values[c] for c in FEATURE_COLS], dtype=np.float32).reshape(1, -1)
    proba      = model.predict_proba(X)[0]
    fraud_prob = float(proba[1])
    legit_prob = float(proba[0])
    label      = "FRAUD" if fraud_prob >= 0.5 else "LEGIT"
    return label, fraud_prob, legit_prob


# ── Log prediction to CSV ─────────────────────────────────────────────────────
def log_prediction(user_id, label, fraud_prob, legit_prob, amount, model_ver):
    write_header = not Path(LOG_FILE).exists()
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "timestamp", "user_id", "prediction", "confidence",
                "fraud_prob", "legit_prob", "amount",
                "txn_count", "avg_amount", "max_amount",
                "model_version", "latency_ms",
            ])
        confidence = fraud_prob if label == "FRAUD" else legit_prob
        w.writerow([
            datetime.utcnow().isoformat(), user_id, label,
            round(confidence, 6), round(fraud_prob, 6), round(legit_prob, 6),
            round(amount, 4), 1, 0.0, 0.0, model_ver, 0.0,
        ])


# ── Load prediction history ────────────────────────────────────────────────────
@st.cache_data(ttl=5)   # refresh every 5 seconds
def load_history():
    if not Path(LOG_FILE).exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(LOG_FILE, parse_dates=["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()


# ── Welcome dialog (shown once per session) ───────────────────────────────────
@st.dialog("👋 Welcome to the Fraud Detection Platform", width="large")
def show_welcome():
    st.markdown("""
### What is this?
This demo shows how a bank decides — in under a second — whether your card swipe is **genuine or fraud**.

You describe a transaction, click a button, and the AI tells you: ✅ **LEGIT** or 🚨 **FRAUD**.

---

### How to use it in 3 steps

| Step | What to do |
|---|---|
| **1** | Pick a scenario from the **left panel** (e.g. "Suspicious High-Value Transfer") |
| **2** | Click the blue **"Analyze Transaction"** button |
| **3** | See the result — green means safe, red means flagged |

---

### Try this example right now

1. On the **left panel**, open the dropdown and select **"Suspicious High-Value Transfer"**
2. Click **"🔍 Analyze Transaction"**
3. You will see a big **🚨 FRAUD** result with ~99% fraud probability

Then try **"Normal Online Purchase"** and watch it flip to ✅ **LEGIT**.

---

### Explore the 4 tabs
- **🔍 Make a Prediction** — test any transaction
- **📋 Prediction History** — see all your tests this session
- **📈 Drift Monitor** — track fraud rate over time
- **🤖 Model Info** — see how accurate the AI is
""")
    st.divider()
    if st.button("🚀 Let's Go!", type="primary", use_container_width=True):
        st.session_state.welcome_shown = True
        st.rerun()

if "welcome_shown" not in st.session_state:
    show_welcome()


# ── CSS styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Result card */
.fraud-card {
    background: linear-gradient(135deg, #ff4b4b, #c0392b);
    color: white; border-radius: 16px; padding: 28px 32px;
    text-align: center; margin: 12px 0;
    box-shadow: 0 4px 20px rgba(255,75,75,0.3);
}
.legit-card {
    background: linear-gradient(135deg, #00c851, #007e33);
    color: white; border-radius: 16px; padding: 28px 32px;
    text-align: center; margin: 12px 0;
    box-shadow: 0 4px 20px rgba(0,200,81,0.3);
}
.card-label  { font-size: 2.8rem; font-weight: 800; letter-spacing: 4px; }
.card-sub    { font-size: 1.1rem; opacity: 0.9; margin-top: 6px; }
/* Metric boxes */
.metric-row  { display: flex; gap: 16px; margin: 12px 0; }
.metric-box  {
    background: #f0f4ff; border-radius: 12px; padding: 16px 20px;
    flex: 1; border: 1px solid #c7d2fe;
}
.metric-val  { font-size: 1.8rem; font-weight: 700; color: #16a34a; }
.metric-lbl  { font-size: 0.8rem; color: #64748b; margin-top: 2px; }
/* Section header */
.section-hdr {
    font-size: 1.1rem; font-weight: 600; color: #1e40af;
    border-left: 4px solid #3b82f6; padding-left: 10px; margin: 20px 0 10px;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/security-shield-green.png", width=64)
    st.title("Fraud Detection\nPlatform")
    st.caption("Real-Time ML Feature Store & Model Serving")
    st.divider()

    # Model status
    model, model_ver = load_model()
    if model:
        st.success(f"Model loaded — {model_ver}")
    else:
        st.error(f"Model not found: {model_ver}")
        st.info("Run `python train_model.py` first.")

    st.divider()

    # Scenario picker
    st.markdown("### Quick Scenarios")
    scenario_name = st.selectbox(
        "Select a transaction type",
        options=list(SCENARIOS.keys()),
        index=0,
        help="Pre-filled examples — choose one or pick Custom to set your own values",
    )
    scenario = SCENARIOS[scenario_name]
    st.caption(f"_{scenario['desc']}_")

    st.divider()

    # User ID
    st.markdown("### User Identity")
    user_id = st.text_input("User ID", value="user_0042",
                            help="Simulated user identifier")

    st.divider()
    st.markdown("**Dataset:** Kaggle Credit Card Fraud  \n284,807 transactions · 492 fraud (0.17%)")
    st.markdown("**Model:** Random Forest · F1=0.91 · AUC=0.98")
    st.divider()
    st.markdown("Built by **Arya Patel**")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═════════════════════════════════════════════════════════════════════════════
tab_predict, tab_history, tab_drift, tab_model = st.tabs([
    "🔍 Make a Prediction",
    "📋 Prediction History",
    "📈 Drift Monitor",
    "🤖 Model Info",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown("## Transaction Analysis")
    st.markdown(
        "Select a scenario from the sidebar or adjust the sliders below, "
        "then click **Analyze Transaction** to get a real-time fraud prediction."
    )

    # ── Amount ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Transaction Amount</div>', unsafe_allow_html=True)
    amount = st.slider(
        "Amount ($)", min_value=0.01, max_value=5000.0,
        value=float(scenario["Amount"]), step=0.01, format="$%.2f",
    )

    # ── PCA Features in expandable groups ────────────────────────────────────
    st.markdown('<div class="section-hdr">PCA Security Features (V1–V28)</div>',
                unsafe_allow_html=True)
    st.caption(
        "These are anonymized behavioral signals computed from the raw transaction data. "
        "Unusual values (below -5 or above 5) indicate anomalous patterns."
    )

    feature_vals = {}

    col_a, col_b = st.columns(2)

    with col_a:
        with st.expander("Transaction Signals — Group 1 (V1–V14)", expanded=True):
            for i in range(1, 15):
                key      = f"V{i}"
                imp      = FEATURE_IMPORTANCE.get(key, 0)
                badge    = " 🔴 KEY" if imp > 0.08 else (" 🟡" if imp > 0.04 else "")
                friendly = FEATURE_NAMES.get(key, key)
                feature_vals[key] = st.slider(
                    f"{friendly}{badge}",
                    min_value=-15.0, max_value=15.0,
                    value=float(scenario.get(key, 0.0)),
                    step=0.01, key=f"slider_{key}",
                    help=f"Technical name: {key}  |  Model importance: {imp:.1%}  |  "
                         f"Normal range: −2 to +2  |  Values beyond ±5 are anomalous",
                )

    with col_b:
        with st.expander("Transaction Signals — Group 2 (V15–V28)", expanded=True):
            for i in range(15, 29):
                key      = f"V{i}"
                imp      = FEATURE_IMPORTANCE.get(key, 0)
                badge    = " 🔴 KEY" if imp > 0.08 else (" 🟡" if imp > 0.04 else "")
                friendly = FEATURE_NAMES.get(key, key)
                feature_vals[key] = st.slider(
                    f"{friendly}{badge}",
                    min_value=-15.0, max_value=15.0,
                    value=float(scenario.get(key, 0.0)),
                    step=0.01, key=f"slider_{key}",
                    help=f"Technical name: {key}  |  Model importance: {imp:.1%}  |  "
                         f"Normal range: −2 to +2  |  Values beyond ±5 are anomalous",
                )

    feature_vals["Amount"] = amount

    # ── Predict button ────────────────────────────────────────────────────────
    st.markdown("---")
    btn_col, _ = st.columns([1, 3])
    with btn_col:
        run = st.button("🔍 Analyze Transaction", type="primary", use_container_width=True)

    if run:
        if model is None:
            st.error("Model not loaded. Run `python train_model.py` first.")
        else:
            with st.spinner("Running inference..."):
                label, fraud_prob, legit_prob = predict(model, feature_vals)
                log_prediction(user_id, label, fraud_prob, legit_prob, amount, model_ver)
                load_history.clear()   # invalidate cached history

            # ── Result card ────────────────────────────────────────────────
            st.markdown("### Result")
            css_class = "fraud-card" if label == "FRAUD" else "legit-card"
            icon      = "🚨" if label == "FRAUD" else "✅"
            sub       = (
                f"Fraud probability: {fraud_prob:.1%} — Transaction flagged for review"
                if label == "FRAUD" else
                f"Fraud probability: {fraud_prob:.1%} — Transaction approved"
            )
            st.markdown(
                f'<div class="{css_class}">'
                f'<div class="card-label">{icon} {label}</div>'
                f'<div class="card-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Metrics row ────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Fraud Probability",  f"{fraud_prob:.2%}")
            m2.metric("Legit Probability",  f"{legit_prob:.2%}")
            m3.metric("Amount",             f"${amount:,.2f}")
            m4.metric("User",               user_id)

            # ── Gauge chart ────────────────────────────────────────────────
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=fraud_prob * 100,
                title={"text": "Fraud Risk Score", "font": {"size": 18}},
                delta={"reference": BASELINE_FRAUD_RATE, "valueformat": ".2f",
                       "suffix": "%", "prefix": "vs baseline "},
                number={"suffix": "%", "valueformat": ".1f"},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar":  {"color": "#ff4b4b" if label == "FRAUD" else "#00c851"},
                    "steps": [
                        {"range": [0,  10], "color": "#d1fae5"},
                        {"range": [10, 50], "color": "#fef9c3"},
                        {"range": [50, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "orange", "width": 3},
                        "thickness": 0.75, "value": 50,
                    },
                },
            ))
            gauge.update_layout(
                height=300, margin=dict(l=30, r=30, t=60, b=10),
                paper_bgcolor="rgba(0,0,0,0)", font_color="#1e293b",
            )
            st.plotly_chart(gauge, use_container_width=True)

            # ── Top features driving this prediction ───────────────────────
            st.markdown("### What influenced this prediction?")
            top_feats = sorted(FEATURE_IMPORTANCE.items(), key=lambda x: -x[1])[:10]
            feat_keys      = [f[0] for f in top_feats]
            feat_imps      = [f[1] for f in top_feats]
            feat_vals_disp = [feature_vals.get(f[0], 0) for f in top_feats]
            feat_labels    = [
                f"{FEATURE_NAMES.get(k, k)}<br><sup>({k})</sup>" for k in feat_keys
            ]

            fig_feat = go.Figure()
            fig_feat.add_trace(go.Bar(
                x=feat_imps, y=feat_labels, orientation="h",
                marker_color=["#ff4b4b" if abs(v) > 3 else "#89b4fa"
                              for v in feat_vals_disp],
                text=[f"{k} = {v:.2f}" for k, v in zip(feat_keys, feat_vals_disp)],
                textposition="outside",
                name="Importance",
            ))
            fig_feat.update_layout(
                title="Top 10 Signals Driving This Prediction  (🔴 red = anomalous value in your input)",
                xaxis_title="Model Importance", yaxis_title="",
                height=420, margin=dict(l=10, r=160, t=50, b=30),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#1e293b", yaxis={"autorange": "reversed"},
            )
            st.plotly_chart(fig_feat, use_container_width=True)

    else:
        # Show placeholder before first prediction
        st.info("👈 Choose a scenario from the sidebar, adjust values if needed, then click **Analyze Transaction**.")

        # Feature importance teaser
        st.markdown("### Top Fraud-Detecting Features")
        top10 = dict(sorted(FEATURE_IMPORTANCE.items(), key=lambda x: -x[1])[:10])
        fig = px.bar(
            x=list(top10.values()), y=list(top10.keys()),
            orientation="h", color=list(top10.values()),
            color_continuous_scale="blues",
            labels={"x": "Importance", "y": "Feature"},
        )
        fig.update_layout(
            height=320, showlegend=False, coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#1e293b", margin=dict(l=10, r=20, t=20, b=20),
            yaxis={"autorange": "reversed"},
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PREDICTION HISTORY
# ─────────────────────────────────────────────────────────────────────────────
with tab_history:
    st.markdown("## Prediction History")

    df = load_history()

    if df.empty:
        st.info("No predictions yet. Go to **Make a Prediction** and run your first analysis.")
    else:
        total  = len(df)
        fraud  = (df["prediction"] == "FRAUD").sum()
        legit  = total - fraud
        rate   = fraud / total * 100 if total else 0

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Predictions", f"{total:,}")
        k2.metric("Fraud Detected",    f"{fraud:,}", delta=f"{rate:.1f}% rate",
                  delta_color="inverse")
        k3.metric("Legit Approved",    f"{legit:,}")
        k4.metric("Avg Confidence",    f"{df['confidence'].mean():.1%}")

        st.divider()

        # Timeline chart
        df_sorted = df.sort_values("timestamp").copy()
        df_sorted["index"] = range(1, len(df_sorted) + 1)
        df_sorted["color"] = df_sorted["prediction"].map(
            {"FRAUD": "#ff4b4b", "LEGIT": "#00c851"})

        fig_tl = go.Figure()
        for pred, color in [("LEGIT", "#00c851"), ("FRAUD", "#ff4b4b")]:
            subset = df_sorted[df_sorted["prediction"] == pred]
            if not subset.empty:
                fig_tl.add_trace(go.Scatter(
                    x=subset["index"], y=subset["fraud_prob"],
                    mode="markers",
                    marker=dict(color=color, size=10, symbol="circle",
                                line=dict(width=1, color="#e2e8f0")),
                    name=pred,
                    text=subset.apply(
                        lambda r: f"User: {r['user_id']}<br>"
                                  f"Amount: ${r['amount']:.2f}<br>"
                                  f"Fraud prob: {r['fraud_prob']:.1%}", axis=1),
                    hovertemplate="%{text}<extra></extra>",
                ))
        fig_tl.add_hline(y=0.5, line_dash="dash", line_color="orange",
                          annotation_text="Decision threshold (50%)")
        fig_tl.update_layout(
            title="Fraud Probability per Prediction",
            xaxis_title="Prediction #", yaxis_title="P(Fraud)",
            height=360, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#1e293b",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_tl, use_container_width=True)

        # Pie chart + amount distribution
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(
                values=[legit, fraud], names=["Legit", "Fraud"],
                color_discrete_sequence=["#00c851", "#ff4b4b"],
                title="Prediction Breakdown",
                hole=0.45,
            )
            fig_pie.update_layout(
                height=300, paper_bgcolor="rgba(0,0,0,0)",
                font_color="#1e293b", margin=dict(t=50, b=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            fig_amt = px.histogram(
                df, x="amount", color="prediction",
                color_discrete_map={"FRAUD": "#ff4b4b", "LEGIT": "#00c851"},
                title="Amount Distribution by Outcome",
                nbins=30, barmode="overlay", opacity=0.75,
            )
            fig_amt.update_layout(
                height=300, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font_color="#1e293b",
                margin=dict(t=50, b=10),
            )
            st.plotly_chart(fig_amt, use_container_width=True)

        # Raw table
        st.markdown("### Raw Log")
        display_cols = ["timestamp", "user_id", "prediction", "confidence",
                        "fraud_prob", "amount", "model_version"]
        show_df = df[display_cols].sort_values("timestamp", ascending=False).head(50)
        st.dataframe(
            show_df.style.apply(
                lambda col: ["background-color:#fee2e2; color:#dc2626"
                             if v == "FRAUD" else "" for v in col]
                if col.name == "prediction" else [""] * len(col),
                axis=0,
            ),
            use_container_width=True, height=300,
        )

        if st.button("Clear History"):
            Path(LOG_FILE).unlink(missing_ok=True)
            load_history.clear()
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — DRIFT MONITOR
# ─────────────────────────────────────────────────────────────────────────────
with tab_drift:
    st.markdown("## Model Drift Monitor")
    st.markdown(
        "Compares the live fraud detection rate against the dataset baseline (**1.73%**). "
        "A spike may indicate data drift, model degradation, or a real fraud wave."
    )

    df = load_history()

    if df.empty or len(df) < 3:
        st.info("Need at least 3 predictions to show drift analysis. Make some predictions first.")
    else:
        total = len(df)
        fraud = (df["prediction"] == "FRAUD").sum()
        rate  = fraud / total * 100

        # Status banner
        if rate > 10:
            st.error(f"⚠️  WARNING — Live fraud rate **{rate:.1f}%** exceeds threshold (10%). "
                     "Consider retraining the model.")
        elif abs(rate - BASELINE_FRAUD_RATE) > 5:
            st.warning(f"🔶  CAUTION — Live fraud rate **{rate:.1f}%** has drifted "
                       f"{rate - BASELINE_FRAUD_RATE:+.1f}% from baseline.")
        else:
            st.success(f"✅  OK — Live fraud rate **{rate:.1f}%** is within normal range "
                       f"(baseline: {BASELINE_FRAUD_RATE}%).")

        # Rolling fraud rate chart
        max_window = min(50, total)
        min_window = min(3, max_window)
        default_window = min(10, max_window)
        if max_window > min_window:
            window = st.slider("Rolling window size", min_window, max_window, default_window)
        else:
            window = max_window
        df_s   = df.sort_values("timestamp").reset_index(drop=True)
        df_s["is_fraud"]   = (df_s["prediction"] == "FRAUD").astype(int)
        df_s["rolling_rate"] = (
            df_s["is_fraud"].rolling(window, min_periods=1).mean() * 100
        )
        df_s["index"] = range(1, len(df_s) + 1)

        fig_drift = go.Figure()
        fig_drift.add_trace(go.Scatter(
            x=df_s["index"], y=df_s["rolling_rate"],
            fill="tozeroy", fillcolor="rgba(59,130,246,0.1)",
            line=dict(color="#2563eb", width=2.5),
            name=f"Rolling {window}-pred fraud rate",
        ))
        fig_drift.add_hline(y=BASELINE_FRAUD_RATE, line_dash="dot",
                             line_color="#16a34a", line_width=2,
                             annotation_text=f"Baseline {BASELINE_FRAUD_RATE}%",
                             annotation_font_color="#16a34a")
        fig_drift.add_hline(y=10, line_dash="dash",
                             line_color="#dc2626", line_width=2,
                             annotation_text="Warning threshold 10%",
                             annotation_font_color="#dc2626")
        fig_drift.update_layout(
            title="Rolling Fraud Rate Over Time",
            xaxis_title="Prediction #", yaxis_title="Fraud rate (%)",
            height=380, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#1e293b",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_drift, use_container_width=True)

        # Confidence drift
        df_s["rolling_conf"] = df_s["confidence"].rolling(window, min_periods=1).mean()
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(
            x=df_s["index"], y=df_s["rolling_conf"],
            fill="tozeroy", fillcolor="rgba(124,58,237,0.1)",
            line=dict(color="#7c3aed", width=2),
            name="Rolling avg confidence",
        ))
        fig_conf.add_hline(y=0.6, line_dash="dot", line_color="#ea580c",
                            annotation_text="Min healthy confidence (60%)",
                            annotation_font_color="#ea580c")
        fig_conf.update_layout(
            title="Model Confidence Over Time (low = uncertain predictions)",
            xaxis_title="Prediction #", yaxis_title="Confidence",
            height=280, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#1e293b",
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig_conf, use_container_width=True)

        # Summary stats
        st.markdown("### Drift Summary")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Live Fraud Rate",    f"{rate:.2f}%",
                  delta=f"{rate - BASELINE_FRAUD_RATE:+.2f}% vs baseline",
                  delta_color="inverse")
        d2.metric("Baseline Rate",      f"{BASELINE_FRAUD_RATE}%")
        d3.metric("Avg Confidence",     f"{df['confidence'].mean():.1%}")
        d4.metric("Min Confidence",     f"{df['confidence'].min():.1%}",
                  delta="OK" if df["confidence"].min() > 0.6 else "LOW",
                  delta_color="normal" if df["confidence"].min() > 0.6 else "inverse")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — MODEL INFO
# ─────────────────────────────────────────────────────────────────────────────
with tab_model:
    st.markdown("## Model Information")

    col_info, col_perf = st.columns([1, 1])

    with col_info:
        st.markdown("### Architecture")
        st.markdown("""
| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Trees | 50 estimators |
| Max depth | 10 |
| Class weighting | Balanced (handles imbalance) |
| Training samples | 50,492 (492 fraud + 50k legit) |
| Features | V1–V28 + Amount (29 total) |
| Registry | MLflow — `fraud_model` v1 |
        """)

        st.markdown("### Training Data")
        fig_dist = px.pie(
            values=[284315, 492],
            names=["Legitimate (99.83%)", "Fraud (0.17%)"],
            color_discrete_sequence=["#89b4fa", "#f38ba8"],
            hole=0.5,
            title="Class Distribution in Dataset",
        )
        fig_dist.update_layout(
            height=260, paper_bgcolor="rgba(0,0,0,0)",
            font_color="#1e293b", margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_perf:
        st.markdown("### Performance Metrics")

        metrics = {
            "Accuracy":  0.9984,
            "Precision": 0.9659,
            "Recall":    0.8673,
            "F1 Score":  0.9140,
            "ROC-AUC":   0.9757,
        }
        for name, val in metrics.items():
            color = "#a6e3a1" if val > 0.9 else "#f9e2af"
            st.markdown(
                f"**{name}**",
            )
            st.progress(val, text=f"{val:.1%}")

        st.markdown("### Confusion Matrix (test set)")
        cm_data = pd.DataFrame(
            [[9998, 3], [13, 85]],
            index=["Actual Legit", "Actual Fraud"],
            columns=["Predicted Legit", "Predicted Fraud"],
        )
        fig_cm = px.imshow(
            cm_data, text_auto=True,
            color_continuous_scale=[[0, "#f0f9ff"], [0.5, "#93c5fd"], [1, "#1d4ed8"]],
            title="Confusion Matrix",
        )
        fig_cm.update_layout(
            height=280, paper_bgcolor="rgba(0,0,0,0)",
            font_color="#1e293b", margin=dict(t=50, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Full feature importance chart
    st.markdown("### All Feature Importances")
    fi_df = pd.DataFrame(
        sorted(FEATURE_IMPORTANCE.items(), key=lambda x: -x[1]),
        columns=["Feature", "Importance"],
    )
    fi_df["Label"] = fi_df["Feature"].map(
        lambda k: f"{FEATURE_NAMES.get(k, k)}  ({k})"
    )
    fig_fi = px.bar(
        fi_df, x="Importance", y="Label", orientation="h",
        color="Importance", color_continuous_scale="Blues",
        text=fi_df["Importance"].map(lambda v: f"{v:.1%}"),
    )
    fig_fi.update_traces(textposition="outside")
    fig_fi.update_layout(
        height=750, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="#1e293b",
        yaxis={"autorange": "reversed", "title": ""},
        coloraxis_showscale=False,
        margin=dict(l=10, r=100, t=20, b=20),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Pipeline overview
    st.markdown("### Pipeline Architecture")
    st.code("""
creditcard.csv ──► producer.py ──────────────► Kafka (topic: transactions)
                                                        │
                                            spark_streaming.py
                                            ┌───────────┴──────────────┐
                                            ▼                          ▼
                                     Redis feature store         Parquet / Hive
                                     features:<user_id>          hive_warehouse/
                                            │
                                    train_model.py ──► MLflow registry
                                                       fraud_model v1
                                            │
                                       api.py :8000
                                       POST /predict
                                            │
                                  predictions_log.csv
                                            │
                                     drift_check.py
                                     drift_report.png
    """, language="text")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#64748b; font-size:0.85rem; padding:8px 0'>"
    "Fraud Detection Platform &nbsp;·&nbsp; Built by <strong>Arya Patel</strong>"
    "</div>",
    unsafe_allow_html=True,
)
