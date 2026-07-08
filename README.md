# FraudShield AI — Real-Time Fraud Detection Engine

An AI-powered real-time fraud detection platform for financial institutions,
built with XGBoost and trained on 50,000 synthetic transactions — **100%
local, no cloud ML API required.**

## Verified performance (on synthetic test set)

| Metric | Value |
|---|---|
| AUC | 1.0 |
| Precision | 99.34% |
| Recall | 100% |
| F1 | 99.67% |

> These numbers reflect performance on synthetic, clearly-separable fraud
> patterns generated for this demo — not real-world transaction data. Real
> fraud detection is a much harder, noisier problem; treat these metrics as
> a demonstration of the pipeline, not a benchmark of production-grade
> real-world fraud detection.

## Project structure

```
fraudshield-ai/
├── data/
│   └── generate_data.py       # synthetic transaction generator (50k transactions, 3% fraud rate)
├── core/
│   └── fraud_detection.py     # XGBoost model, scoring, batch scoring, explainability
├── dashboard/
│   └── app.py                 # Streamlit + Plotly dashboard
├── db/
│   └── fraudshield.db         # SQLite database (generated)
├── requirements.txt
└── README.md
```

## Setup

```bash
cd fraudshield-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 1 — Generate synthetic transaction data

```bash
python3 data/generate_data.py
```

Generates 50,000 transactions across 2,000 accounts, with 4 distinct
synthetic fraud patterns (high-amount, odd-hour, rapid-fire, foreign-no-card).

## Step 2 — Run the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501` with 3 pages:

1. **Overview** — model performance metrics, score distribution, risk breakdown
2. **Live Transaction Feed** — simulated real-time scoring of sampled transactions
3. **Score a Transaction** — manually enter transaction details and get a real-time fraud score with explainability

## Running the model standalone

```bash
python3 -m core.fraud_detection
```

## Notes

- All transaction data is synthetically generated (seeded, `np.random.seed(42)`)
  — there is no real financial data involved anywhere in this project.
- Explainability uses XGBoost's built-in feature importances (global, not
  per-prediction SHAP values) — described in the dashboard as "top contributing
  factors," not a full SHAP explanation.
- Model retrains in-memory on each app restart (not persisted to disk) — this
  keeps the demo simple but means restarting the app takes a few seconds to
  retrain before serving predictions.

## License

MIT — see [LICENSE](LICENSE).
