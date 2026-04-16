import pandas as pd
import numpy as np
import joblib

# Load model + scaler
model = joblib.load("models/model.pkl")

df = pd.read_csv("data/featured_games.csv")

# Use latest season
latest_season = df["SEASON_home"].max()
df = df[df["SEASON_home"] == latest_season].copy()


def build_team_state(df):
    """
    Builds a clean state dictionary for each team using latest known stats
    """
    state = {}

    teams = pd.concat([df["home_team"], df["away_team"]]).unique()

    for team in teams:
        # Get all games involving team
        team_games = df[(df["home_team"] == team) | (df["away_team"] == team)]

        # Sort properly
        team_games = team_games.sort_values("date")

        latest = team_games.iloc[-1]

        if latest["home_team"] == team:
            state[team] = {
                "win_pct": latest["home_win_pct"],
                "recent": latest["home_recent_win_pct"],
                "net": latest["home_net_rating"]
            }
        else:
            state[team] = {
                "win_pct": latest["away_win_pct"],
                "recent": latest["away_recent_win_pct"],
                "net": latest["away_net_rating"]
            }

    return state


def predict_game(state, team_home, team_away):
    home = state[team_home]
    away = state[team_away]

    features = pd.DataFrame([{
        "win_pct_diff": home["win_pct"] - away["win_pct"],
        "recent_win_pct_diff": home["recent"] - away["recent"],
        "net_rating_diff": home["net"] - away["net"]
    }])

    prob = model.predict_proba(features)[0][1]

    return prob

state = build_team_state(df)

okc = state["OKC"]
was = state["WAS"]

print("OKC:", okc)
print("WAS:", was)

print("DIFFS:")
print("win_pct_diff:", okc["win_pct"] - was["win_pct"])
print("recent_diff:", okc["recent"] - was["recent"])
print("net_diff:", okc["net"] - was["net"])