import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from core.fraud_detection import train_model, get_metrics, score_transaction, score_batch

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "fraudshield.db")

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")


def kpi_card(label, value, delta=None):
    st.metric(label, value, delta)


with st.sidebar:
    st.title("🛡️ FraudShield AI")
    st.caption("Real-time fraud detection engine")
    st.markdown("---")
    page = st.radio("Navigate", ["Overview", "Live Transaction Feed", "Score a Transaction"])
    st.markdown("---")
    st.caption("⚠️ Trained on synthetic transaction data for demo purposes.")

with st.spinner("Training fraud detection model..."):
    metrics = get_metrics(DB_PATH)

if page == "Overview":
    st.title("Fraud Detection Model Overview")
    st.caption("XGBoost classifier trained on 50,000 synthetic transactions")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("AUC", metrics["auc"])
    with col2:
        kpi_card("Precision", f"{metrics['precision']*100:.2f}%")
    with col3:
        kpi_card("Recall", f"{metrics['recall']*100:.2f}%")
    with col4:
        kpi_card("F1 Score", f"{metrics['f1']*100:.2f}%")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        kpi_card("Training Set Size", f"{metrics['n_train']:,}")
    with col_b:
        kpi_card("Fraud Rate in Data", f"{metrics['fraud_rate']*100:.2f}%")

    st.markdown("---")
    st.subheader("Sample Scored Transactions")
    with st.spinner("Scoring sample transactions..."):
        scored = score_batch(DB_PATH, n=200)

    fig = px.histogram(scored, x="fraud_probability", color="risk_level", nbins=30,
                        title="Distribution of Fraud Probability Scores",
                        color_discrete_map={"Low": "#22c55e", "Medium": "#eab308", "High": "#ef4444"})
    st.plotly_chart(fig, use_container_width=True)

    risk_counts = scored["risk_level"].value_counts()
    fig2 = px.pie(values=risk_counts.values, names=risk_counts.index, hole=0.5,
                  title="Risk Level Breakdown (sample)",
                  color=risk_counts.index,
                  color_discrete_map={"Low": "#22c55e", "Medium": "#eab308", "High": "#ef4444"})
    st.plotly_chart(fig2, use_container_width=True)


elif page == "Live Transaction Feed":
    st.title("Live Transaction Scoring Feed")
    st.caption("Simulated real-time feed — sample of real transactions scored by the model")

    n = st.slider("Number of transactions to show", 20, 500, 100)
    with st.spinner("Scoring transactions..."):
        scored = score_batch(DB_PATH, n=n)

    high_risk = (scored["risk_level"] == "High").sum()
    col1, col2, col3 = st.columns(3)
    with col1:
        kpi_card("Transactions Shown", len(scored))
    with col2:
        kpi_card("High Risk Flagged", high_risk)
    with col3:
        kpi_card("Actual Fraud in Sample", int(scored["is_fraud"].sum()))

    display_cols = ["transaction_id", "account_id", "amount", "merchant_category",
                     "hour", "is_foreign", "card_present", "fraud_probability", "risk_level", "is_fraud"]
    st.dataframe(scored[display_cols].rename(columns={"is_fraud": "actual_fraud"}),
                 use_container_width=True, height=500)


elif page == "Score a Transaction":
    st.title("Score a Transaction")
    st.caption("Enter transaction details to get a real-time fraud risk score")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount ($)", min_value=0.0, value=150.0, step=10.0)
        hour = st.slider("Hour of day", 0, 23, 14)
        merchant = st.selectbox("Merchant category",
                                 ["Grocery", "Electronics", "Travel", "Restaurant",
                                  "Online Retail", "Gas Station", "Utilities", "ATM Withdrawal"])
        is_weekend = st.checkbox("Weekend transaction")
    with col2:
        distance = st.number_input("Distance from home (km)", min_value=0.0, value=5.0, step=1.0)
        recent_txns = st.number_input("Transactions in last hour", min_value=0, value=0, step=1)
        is_foreign = st.checkbox("Foreign transaction")
        card_present = st.checkbox("Card present (in-person)", value=True)

    if st.button("Score Transaction", type="primary"):
        txn = {
            "amount": amount, "hour": hour, "is_weekend": int(is_weekend),
            "distance_from_home_km": distance, "transactions_last_hour": recent_txns,
            "is_foreign": int(is_foreign), "card_present": int(card_present),
            "merchant_category": merchant,
        }
        result = score_transaction(txn, DB_PATH)

        color = {"Low": "green", "Medium": "orange", "High": "red"}
        st.markdown(f"### Risk Level: :{color[result['risk_level']]}[{result['risk_level']}]")
        st.markdown(f"**Fraud Probability:** {result['fraud_probability']*100:.2f}%")

        st.subheader("Top Contributing Factors")
        for factor in result["top_factors"]:
            st.markdown(f"- **{factor['feature']}** — importance: {factor['importance']:.3f}")
