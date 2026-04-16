import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler


def train_model(df):

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # -----------------------------
    # TARGET
    # -----------------------------
    df["home_win"] = df["home_win"].astype(int)

    # -----------------------------
    # FEATURES (THIS MUST MATCH simulate.py)
    # -----------------------------
    features = [
    "elo_diff",              # main strength
    "matchup_diff",          # style
    "recent_win_pct_diff"    # momentum
    ]

    df = df.dropna(subset=features)

    # -----------------------------
    # TIME SPLIT
    # -----------------------------
    split_date = "2024-09-01"

    train_df = df[df["date"] < split_date]
    test_df  = df[df["date"] >= split_date]

    X_train = train_df[features]
    y_train = train_df["home_win"]

    X_test = test_df[features]
    y_test = test_df["home_win"]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # MODEL 
    # -----------------------------
    model = LogisticRegression(
    max_iter=1000,
    C=0.5
    )

    model.fit(X_train, y_train)

# ----------------------------------
# TEST ON HOLDOUT SET (CORRECT WAY)
# ----------------------------------
    prob_pred = model.predict_proba(X_test)[:, 1]

    ll = log_loss(y_test, prob_pred)

    print("\n=== PURE MODEL TEST ===")
    print("Log Loss:", ll)

    # Inspect predictions
    results = pd.DataFrame(X_test.copy())
    results['actual'] = y_test.values
    results["predicted_prob"] = prob_pred
    print(model.coef_)

    print("Test date range:")
    print(test_df["date"].min(), "→", test_df["date"].max())

    # -----------------------------
    # BASELINE (IMPORTANT)
    # -----------------------------
    mask = (
        (df["win_pct_diff"].abs() < 0.02) &
        (df["recent_win_pct_diff"].abs() < 0.02)
    )

    baseline = df["home_win"][mask].mean()

    print("Equal-team baseline:", baseline)

    # -----------------------------
    # SAVE
    # -----------------------------
    joblib.dump(model, "models/model.pkl")
    joblib.dump(baseline, "models/baseline.pkl")
    joblib.dump(features, "models/features.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    return model, baseline


if __name__ == "__main__":
    df = pd.read_csv("data/featured_games.csv")
    train_model(df)