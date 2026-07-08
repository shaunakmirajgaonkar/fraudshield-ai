"""
Real-time fraud detection engine - XGBoost classifier with explainable
feature-importance scoring. 100% local, no cloud ML API.
"""
import sqlite3
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

FEATURE_COLS = ["amount", "hour", "is_weekend", "distance_from_home_km",
                "transactions_last_hour", "is_foreign", "card_present", "merchant_encoded"]

_model = None
_merchant_encoder = None
_metrics = None


def _load_data(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()
    return df


def train_model(db_path: str = "db/fraudshield.db"):
    """Train the XGBoost fraud classifier and cache it in memory."""
    global _model, _merchant_encoder, _metrics

    df = _load_data(db_path)
    _merchant_encoder = LabelEncoder()
    df["merchant_encoded"] = _merchant_encoder.fit_transform(df["merchant_category"])

    X = df[FEATURE_COLS]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    _metrics = {
        "auc": round(roc_auc_score(y_test, proba), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "fraud_rate": round(y.mean(), 4),
    }
    _model = model
    return _metrics


def get_metrics(db_path: str = "db/fraudshield.db") -> dict:
    if _metrics is None:
        train_model(db_path)
    return _metrics


def score_transaction(transaction: dict, db_path: str = "db/fraudshield.db") -> dict:
    """Score a single transaction and return fraud probability + explainability."""
    global _model, _merchant_encoder
    if _model is None:
        train_model(db_path)

    merchant = transaction.get("merchant_category", "Grocery")
    try:
        merchant_encoded = _merchant_encoder.transform([merchant])[0]
    except ValueError:
        merchant_encoded = 0

    row = pd.DataFrame([{
        "amount": transaction["amount"],
        "hour": transaction["hour"],
        "is_weekend": transaction["is_weekend"],
        "distance_from_home_km": transaction["distance_from_home_km"],
        "transactions_last_hour": transaction["transactions_last_hour"],
        "is_foreign": transaction["is_foreign"],
        "card_present": transaction["card_present"],
        "merchant_encoded": merchant_encoded,
    }])

    proba = float(_model.predict_proba(row)[0, 1])
    importances = _model.feature_importances_
    feature_contributions = sorted(
        zip(FEATURE_COLS, importances), key=lambda x: -x[1]
    )[:3]

    risk_level = "High" if proba >= 0.7 else "Medium" if proba >= 0.3 else "Low"

    return {
        "fraud_probability": round(proba, 4),
        "risk_level": risk_level,
        "top_factors": [{"feature": f, "importance": round(float(i), 4)} for f, i in feature_contributions],
    }


def score_batch(db_path: str = "db/fraudshield.db", n: int = 500) -> pd.DataFrame:
    """Score a sample of real transactions from the DB for the dashboard's live feed."""
    global _model, _merchant_encoder
    if _model is None:
        train_model(db_path)

    df = _load_data(db_path).sample(n=min(n, 5000), random_state=1)
    df["merchant_encoded"] = _merchant_encoder.transform(df["merchant_category"])
    X = df[FEATURE_COLS]
    df["fraud_probability"] = _model.predict_proba(X)[:, 1].round(4)
    df["predicted_fraud"] = (df["fraud_probability"] >= 0.5).astype(int)
    df["risk_level"] = pd.cut(df["fraud_probability"], bins=[-0.01, 0.3, 0.7, 1.01],
                               labels=["Low", "Medium", "High"])
    return df.sort_values("fraud_probability", ascending=False)


if __name__ == "__main__":
    metrics = train_model()
    print(f"AUC: {metrics['auc']}, Precision: {metrics['precision']}, "
          f"Recall: {metrics['recall']}, F1: {metrics['f1']}")

    sample_txn = {"amount": 4500, "hour": 3, "is_weekend": 0, "distance_from_home_km": 3000,
                  "transactions_last_hour": 5, "is_foreign": 1, "card_present": 0,
                  "merchant_category": "Electronics"}
    result = score_transaction(sample_txn)
    print(result)
