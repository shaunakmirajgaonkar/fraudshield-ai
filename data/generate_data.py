"""
Generates 50,000 synthetic financial transactions with planted fraud
patterns for FraudShield AI. 100% local - no external data source.
"""
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

np.random.seed(42)
fake = Faker()
Faker.seed(42)

DB_PATH = "db/fraudshield.db"
N_TRANSACTIONS = 50000
FRAUD_RATE = 0.03

MERCHANT_CATEGORIES = ["Grocery", "Electronics", "Travel", "Restaurant",
                        "Online Retail", "Gas Station", "Utilities", "ATM Withdrawal"]


def generate_transactions(n=N_TRANSACTIONS):
    n_fraud = int(n * FRAUD_RATE)
    n_legit = n - n_fraud

    rows = []
    account_ids = [f"ACC{str(i).zfill(6)}" for i in range(2000)]

    # Legitimate transactions: normal patterns
    for _ in range(n_legit):
        hour = np.random.choice(range(24), p=_hour_weights())
        amount = np.random.lognormal(mean=3.5, sigma=1.0)
        rows.append({
            "account_id": np.random.choice(account_ids),
            "amount": round(min(amount, 5000), 2),
            "merchant_category": np.random.choice(MERCHANT_CATEGORIES),
            "hour": hour,
            "is_weekend": np.random.choice([0, 1], p=[0.71, 0.29]),
            "distance_from_home_km": round(np.random.exponential(15), 1),
            "transactions_last_hour": np.random.poisson(0.3),
            "is_foreign": np.random.choice([0, 1], p=[0.97, 0.03]),
            "card_present": np.random.choice([0, 1], p=[0.3, 0.7]),
            "is_fraud": 0,
        })

    # Fraudulent transactions: distinct suspicious patterns
    for _ in range(n_fraud):
        pattern = np.random.choice(["high_amount", "odd_hour", "rapid_fire", "foreign_no_card"])
        base = {
            "account_id": np.random.choice(account_ids),
            "merchant_category": np.random.choice(MERCHANT_CATEGORIES),
            "card_present": 0,
            "is_fraud": 1,
        }
        if pattern == "high_amount":
            base.update({"amount": round(np.random.uniform(2000, 9000), 2),
                         "hour": np.random.randint(0, 24), "is_weekend": np.random.choice([0, 1]),
                         "distance_from_home_km": round(np.random.exponential(80), 1),
                         "transactions_last_hour": np.random.poisson(1), "is_foreign": np.random.choice([0, 1])})
        elif pattern == "odd_hour":
            base.update({"amount": round(np.random.uniform(200, 3000), 2),
                         "hour": np.random.choice([1, 2, 3, 4]), "is_weekend": np.random.choice([0, 1]),
                         "distance_from_home_km": round(np.random.exponential(50), 1),
                         "transactions_last_hour": np.random.poisson(1), "is_foreign": np.random.choice([0, 1])})
        elif pattern == "rapid_fire":
            base.update({"amount": round(np.random.uniform(50, 1500), 2),
                         "hour": np.random.randint(0, 24), "is_weekend": np.random.choice([0, 1]),
                         "distance_from_home_km": round(np.random.exponential(30), 1),
                         "transactions_last_hour": np.random.poisson(6) + 4, "is_foreign": 0})
        else:  # foreign_no_card
            base.update({"amount": round(np.random.uniform(500, 4000), 2),
                         "hour": np.random.randint(0, 24), "is_weekend": np.random.choice([0, 1]),
                         "distance_from_home_km": round(np.random.uniform(500, 8000), 1),
                         "transactions_last_hour": np.random.poisson(1), "is_foreign": 1})
        rows.append(base)

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

    start = datetime.now() - timedelta(days=90)
    df["timestamp"] = [
        (start + timedelta(days=np.random.randint(0, 90), hours=int(h))).strftime("%Y-%m-%d %H:%M:%S")
        for h in df["hour"]
    ]
    df["transaction_id"] = [f"TXN{str(i).zfill(7)}" for i in range(len(df))]
    return df


def _hour_weights():
    # Legit transactions cluster in daytime/evening hours
    weights = np.array([0.5,0.3,0.2,0.2,0.3,0.5,1,2,3,3.5,3.5,4,4.5,4,3.5,3.5,4,4.5,5,4.5,3.5,2.5,1.5,1])
    return weights / weights.sum()


def main():
    conn = sqlite3.connect(DB_PATH)
    df = generate_transactions()
    df.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print(f"Generated {len(df)} transactions ({df['is_fraud'].sum()} fraudulent, "
          f"{df['is_fraud'].mean()*100:.2f}% fraud rate)")


if __name__ == "__main__":
    main()
