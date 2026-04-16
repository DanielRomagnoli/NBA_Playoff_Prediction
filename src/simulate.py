import numpy as np
import pandas as pd
import joblib
from src.feature_engineering import update_elo

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("models/model.pkl")
baseline = joblib.load("models/baseline.pkl")
features = joblib.load("models/features.pkl")
scaler = joblib.load("models/scaler.pkl")

# -----------------------------
# GLOBAL ELO STATE
# -----------------------------

def update_sim_elo(state, teamA, teamB, result):

    ra = state[teamA]["elo"]
    rb = state[teamB]["elo"]

    K = 20

    ea = 1 / (1 + 10 ** ((rb - ra) / 400))

    state[teamA]["elo"] = ra + K * (result - ea)
    state[teamB]["elo"] = rb + K * ((1 - result) - (1 - ea))


# -----------------------------
# PLAYOFF TEAMS
# -----------------------------
EAST = ["DET","BOS","NYK","CLE","TOR","ATL","PHI","CHA"]
WEST = ["OKC","SAS","DEN","LAL","HOU","MIN","POR","PHX"]


# -----------------------------
# BUILD INITIAL TEAM STATE
# -----------------------------
def build_initial_state(df):

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    latest_season = df["SEASON_home"].max()
    season_df = df[df["SEASON_home"] == latest_season]

    state = {}

    teams = pd.concat([season_df["home_team"], season_df["away_team"]]).unique()

    for team in teams:

        team_games = season_df[
            (season_df["home_team"] == team) |
            (season_df["away_team"] == team)
        ].sort_values("date")

        latest = team_games.iloc[-1]

        if latest["home_team"] == team:
            state[team] = {
                "win_pct": latest["home_win_pct"],
                "recent": latest["home_recent_win_pct"],
                "rolling_off": latest["home_rolling_off"],
                "rolling_def": latest["home_rolling_def"],
                "elo": latest["home_elo"]   
            }
        else:
            state[team] = {
                "win_pct": latest["away_win_pct"],
                "recent": latest["away_recent_win_pct"],
                "rolling_off": latest["away_rolling_off"],
                "rolling_def": latest["away_rolling_def"],
                "elo": latest["away_elo"]   # ✅ KEY
            }

    return state

# -----------------------------
# BUILD SERIES STATE (LIVE PLAYOFFS)
# -----------------------------
def build_series_state(df):

    df = df.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str)

    playoff_games = df[df["GAME_ID"].str.startswith("004")]

    if playoff_games.empty:
        return {}

    series = {}

    for _, row in playoff_games.iterrows():

        home = row["home_team"]
        away = row["away_team"]

        key = tuple(sorted([home, away]))

        if key not in series:
            series[key] = {
                "team_A": key[0],
                "team_B": key[1],
                "wins_A": 0,
                "wins_B": 0,
                "games_played": 0
            }

        winner = home if row["home_win"] == 1 else away

        if winner == series[key]["team_A"]:
            series[key]["wins_A"] += 1
        else:
            series[key]["wins_B"] += 1

        series[key]["games_played"] += 1

    return series

# -----------------------------
# PREDICT GAME
# -----------------------------
def predict_game(state, home, away):

    h = state[home]
    a = state[away]

    X = pd.DataFrame([{
        "elo_diff": h["elo"] - a["elo"],
        "matchup_diff": (h["rolling_off"] - a["rolling_def"]) - (a["rolling_off"] - h["rolling_def"]),
        "recent_win_pct_diff": h["recent"] - a["recent"]
    }])

    X_scaled = scaler.transform(X)

    p = model.predict_proba(X_scaled)[0][1]

    return p
# -----------------------------
# UPDATE STATE AFTER GAME
# -----------------------------
def update_state(state, home, away, home_win):

    if home_win:
        update_sim_elo(state, home, away, 1)
        state[home]["recent"] = min(state[home]["recent"] + 0.01, 1)
        state[away]["recent"] = max(state[away]["recent"] - 0.01, 0)
    else:
        update_sim_elo(state, home, away, 0)
        state[home]["recent"] = max(state[home]["recent"] - 0.01, 0)
        state[away]["recent"] = min(state[away]["recent"] + 0.01, 1)

# -----------------------------
# SIMULATE SERIES
# -----------------------------
def simulate_series(state, team_A, team_B,
                    wins_A=0, wins_B=0, games_played=0):

    home_schedule = [team_A, team_A, team_B, team_B, team_A, team_B, team_A]

    state = {k: v.copy() for k, v in state.items()}

    for i in range(games_played, 7):
        if wins_A == 4 or wins_B == 4:
            break

        home = home_schedule[i]
        away = team_B if home == team_A else team_A

        p = predict_game(state, home, away)
        print(p)

        home_win = np.random.rand() < p
        winner = home if home_win else away

        if winner == team_A:
            wins_A += 1
        else:
            wins_B += 1

        update_state(state, home, away, home_win)

    winner = team_A if wins_A > wins_B else team_B

    return winner, wins_A, wins_B


# -----------------------------
# SERIES WITH STABILITY
# -----------------------------
def simulate_series_n(state, team_A, team_B, n=200):

    outcomes = {}

    for _ in range(n):
        winner, wA, wB = simulate_series(state, team_A, team_B)

        key = (winner, wA, wB)
        outcomes[key] = outcomes.get(key, 0) + 1

    best = max(outcomes, key=outcomes.get)

    return best


# -----------------------------
# FIRST ROUND
# -----------------------------
def first_round(teams):
    return [
        (teams[0], teams[7]),
        (teams[1], teams[6]),
        (teams[2], teams[5]),
        (teams[3], teams[4]),
    ]


# -----------------------------
# SIMULATE ROUND
# -----------------------------
def simulate_round_stable(state, matchups, series_state=None, n=200):

    results = []

    for team_A, team_B in matchups:

        key = tuple(sorted([team_A, team_B]))

        if series_state and key in series_state:
            s = series_state[key]

            if s["team_A"] == team_A:
                wA, wB = s["wins_A"], s["wins_B"]
            else:
                wA, wB = s["wins_B"], s["wins_A"]

            winner, wA, wB = simulate_series(
                state, team_A, team_B, wA, wB, s["games_played"]
            )

        else:
            winner, wA, wB = simulate_series_n(state, team_A, team_B, n)

        results.append({
            "team_A": team_A,
            "team_B": team_B,
            "winner": winner,
            "wins_A": wA,
            "wins_B": wB
        })

    return results


# -----------------------------
# FULL BRACKET
# -----------------------------
def simulate_bracket_stable(state, series_state, n=200):

    bracket = {}

    r1 = simulate_round_stable(state, first_round(EAST), series_state, n)
    r1 += simulate_round_stable(state, first_round(WEST), series_state, n)
    bracket["Round 1"] = r1

    winners_r1 = [s["winner"] for s in r1]

    r2 = simulate_round_stable(state, [
        (winners_r1[0], winners_r1[3]),
        (winners_r1[1], winners_r1[2]),
        (winners_r1[4], winners_r1[7]),
        (winners_r1[5], winners_r1[6]),
    ], None, n)
    bracket["Round 2"] = r2

    winners_r2 = [s["winner"] for s in r2]

    r3 = simulate_round_stable(state, [
        (winners_r2[0], winners_r2[1]),
        (winners_r2[2], winners_r2[3]),
    ], None, n)
    bracket["Conference Finals"] = r3

    winners_r3 = [s["winner"] for s in r3]

    finals = simulate_round_stable(state, [
        (winners_r3[0], winners_r3[1])
    ], None, n)
    bracket["Finals"] = finals

    return bracket


# -----------------------------
# MONTE CARLO
# -----------------------------
def monte_carlo(state, series_state, n=1000):

    import copy
    results = {}

    for _ in range(n):

        state_copy = copy.deepcopy(state)
        series_copy = copy.deepcopy(series_state)

        bracket = simulate_bracket_stable(state_copy, series_copy, 1)

        champion = bracket["Finals"][0]["winner"]
        results[champion] = results.get(champion, 0) + 1

    for team in results:
        results[team] /= n

    return dict(sorted(results.items(), key=lambda x: -x[1]))