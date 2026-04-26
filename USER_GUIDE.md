# Fraud Detection Platform — User Guide

**Live demo:** https://frauddetectionplatformbyaryapatel.streamlit.app

---

## What is this?

This is a **fraud detection demo** that works like the system a bank uses when you swipe your credit card.

Every time you swipe your card at a shop, the bank checks in milliseconds — *"Does this look like a real purchase, or does it look like someone stole this card?"* This demo lets you simulate that check yourself.

You describe a transaction using sliders, click a button, and the system tells you: **LEGIT** (safe) or **FRAUD** (suspicious).

---

## How to open it

1. Open any web browser (Chrome, Safari, Firefox — any works)
2. Go to: **https://frauddetectionplatformbyaryapatel.streamlit.app**
3. Wait about 5–10 seconds for it to load
4. That's it — no login, no download, no installation needed

> On a phone? Just scan the QR code in the project folder (`demo_qr.png`) with your camera.

---

## What you see on screen

The screen has **two parts**:

- **Left side panel (sidebar)** — where you pick what kind of transaction to test
- **Main area** — where you set details and see results

At the top of the main area there are **4 tabs**. Think of them like 4 pages:

| Tab | What it does |
|---|---|
| 🔍 Make a Prediction | Test a transaction right now |
| 📋 Prediction History | See all tests you've done this session |
| 📈 Drift Monitor | See if the fraud rate is changing over time |
| 🤖 Model Info | See how accurate the AI is |

---

## Step-by-step: Making your first prediction

### Step 1 — Pick a scenario (left side panel)

On the left, under **"Quick Scenarios"**, you'll see a dropdown menu. Click it and choose one of these ready-made examples:

| Scenario | What it represents | Expected result |
|---|---|---|
| **Normal Online Purchase** | Buying something on Amazon for $150 | ✅ LEGIT |
| **Small Grocery Purchase** | Buying milk and bread for $10 | ✅ LEGIT |
| **Suspicious High-Value Transfer** | A large, very unusual transaction | 🚨 FRAUD |
| **Overseas Night Transaction** | A charge at 3 AM from another country | 🚨 FRAUD (likely) |
| **Custom (manual input)** | You set every value yourself | Depends on you |

**Tip for beginners:** Start with **"Suspicious High-Value Transfer"** — it will clearly say FRAUD and show you how the result looks.

---

### Step 2 — Set the transaction amount

In the main area, there is a slider called **"Amount ($)"**.

Drag it left (smaller amount) or right (larger amount). The number shows how much money this transaction is for.

> You don't need to change this if you picked a scenario — it's already set to a realistic value.

---

### Step 3 — Look at the other sliders (optional)

Below the amount, you'll see two groups of sliders called:
- **Transaction Signals — Group 1**
- **Transaction Signals — Group 2**

These represent hidden signals the bank collects about every transaction — things like:
- Did the purchase happen far from where the person usually shops?
- Is the amount unusual compared to this person's history?
- Was the card physically present or was it used online?

**You don't need to touch these** if you selected a scenario — they are already filled in correctly.

If you want to experiment, you can move them. Each slider goes from **-15 to +15**. Numbers close to 0 are normal. Numbers below -5 or above +5 are unusual (more suspicious).

The sliders marked **🔴 KEY** are the most important ones the AI looks at. The ones marked **🟡** are moderately important.

---

### Step 4 — Click "Analyze Transaction"

Find the blue button that says **"🔍 Analyze Transaction"** and click it.

The system will think for a second, then show you the result.

---

## Understanding the result

### The big coloured box

After clicking the button, a large coloured box appears:

**Green box that says ✅ LEGIT**
> The AI thinks this is a normal, safe transaction. It would be approved.

**Red box that says 🚨 FRAUD**
> The AI thinks this transaction looks suspicious. It would be flagged for review.

---

### The four numbers below the box

| Number | What it means |
|---|---|
| **Fraud Probability** | Chance (%) the AI thinks this is fraud. Above 50% = flagged as FRAUD |
| **Legit Probability** | Chance (%) the AI thinks this is safe. These two always add up to 100% |
| **Amount** | The transaction amount you set |
| **User** | The fake customer ID being tested |

---

### The gauge (speedometer)

The circular gauge works like a speedometer for fraud risk:
- **Needle pointing left (near 0%)** → very safe
- **Needle pointing right (near 100%)** → very suspicious
- The **orange line** at 50% is the cutoff — anything above that is flagged as FRAUD

---

### The bar chart at the bottom

This shows **which signals mattered most** for this particular prediction.

- **Blue bars** = the signal was in a normal range
- **Red bars** = the signal was unusual (this is what pushed the AI toward saying FRAUD)

---

## Trying different scenarios

Here's a fun way to explore the demo:

1. Pick **"Normal Online Purchase"** → click Analyze → see ✅ LEGIT
2. Pick **"Suspicious High-Value Transfer"** → click Analyze → see 🚨 FRAUD
3. Pick **"Custom"** → drag the **"Location Anomaly Score 🔴 KEY"** slider all the way to **-15** → click Analyze → watch it turn to FRAUD
4. Drag it back to **0** → click Analyze → watch it go back to LEGIT

This shows exactly how one signal can tip the decision.

---

## The other tabs

### 📋 Prediction History

After you make a few predictions, click this tab.

You'll see:
- A count of how many predictions you've made and how many were FRAUD vs LEGIT
- A chart showing each prediction as a dot — green dots are LEGIT, red dots are FRAUD
- A pie chart showing the ratio
- A table with every prediction listed

There is a **"Clear History"** button at the bottom if you want to start fresh.

---

### 📈 Drift Monitor

This tab checks: *"Is the fraud rate in my session higher or lower than the real-world average?"*

The real-world average from the dataset is **1.73%** (about 2 in every 100 transactions are fraud).

- **Green banner (OK)** — your session's fraud rate is close to normal
- **Yellow banner (CAUTION)** — your fraud rate has drifted away from normal
- **Red banner (WARNING)** — your fraud rate is very high (above 10%)

> If you tested only the "Suspicious High-Value Transfer" scenario 10 times, you'll get a WARNING — because 100% of your tests were fraud. That's expected.

You'll also see a chart that plots the fraud rate as you make more predictions.

---

### 🤖 Model Info

This tab shows how the AI was built and how accurate it is.

Key numbers:
| Metric | Value | What it means |
|---|---|---|
| **Accuracy** | 99.84% | Out of 100 transactions, it gets 99.84 correct |
| **Precision** | 96.6% | When it says FRAUD, it's right 96.6% of the time |
| **Recall** | 86.7% | It catches 86.7% of all real fraud cases |
| **F1 Score** | 91.4% | Overall balance between precision and recall |

You'll also see a **Confusion Matrix** — a table that shows exactly how many transactions it got right and wrong during testing.

---

## Common questions

**Q: Is this using real bank data?**
No. The AI was trained on a public dataset of 284,807 anonymised credit card transactions from European cardholders in 2013. No real names, cards, or account numbers are involved.

**Q: Will my predictions be stored forever?**
No. History is saved only for your current browser session. If you close the tab and come back, it starts fresh.

**Q: Why do the sliders have strange names like "Location Anomaly Score"?**
The original data was anonymised by the bank before being made public, so the real signal names were hidden. These friendly names are our best interpretation of what each signal represents.

**Q: The sliders are confusing — can I just use the scenarios?**
Absolutely. Selecting a scenario from the dropdown is all you need. The sliders fill in automatically. Just click "Analyze Transaction" and see the result.

**Q: What does the AI actually use to decide?**
The three most important signals are:
1. **Location Anomaly Score** — did the transaction happen somewhere unusual for this person?
2. **Spending Velocity** — are they spending much faster than usual?
3. **Merchant Category Risk** — is this type of shop associated with fraud?

---

## Quick reference card

```
Open the app  →  frauddetectionplatformbyaryapatel.streamlit.app

Pick scenario (left panel)
       ↓
Click "Analyze Transaction"
       ↓
Green = LEGIT ✅     Red = FRAUD 🚨
       ↓
Check the gauge and bar chart to see why
       ↓
Try another scenario and compare!
```

---

*Built with Python · scikit-learn · FastAPI · Apache Kafka · Redis · MLflow · Streamlit*
