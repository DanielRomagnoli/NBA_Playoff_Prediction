import pandas as pd

def add_team_win_pct(games):
    print("Adding rolling win percentage features...")

    games = games.sort_values(["SEASON_home", "date"])

    team_records = {}
    current_season = None

    home_win_pct = []
    away_win_pct = []

    for _, row in games.iterrows():
        if row["SEASON_home"] != current_season:
            team_records = {}
            current_season = row["SEASON_home"]

        home = row["home_team"]
        away = row["away_team"]

        if home not in team_records:
            team_records[home] = {"wins": 0, "games": 0}
        if away not in team_records:
            team_records[away] = {"wins": 0, "games": 0}

        home_pct = team_records[home]["wins"] / team_records[home]["games"] if team_records[home]["games"] > 0 else 0.5
        away_pct = team_records[away]["wins"] / team_records[away]["games"] if team_records[away]["games"] > 0 else 0.5

        home_win_pct.append(home_pct)
        away_win_pct.append(away_pct)

        if row["home_win"] == 1:
            team_records[home]["wins"] += 1
        else:
            team_records[away]["wins"] += 1
        team_records[home]["games"] += 1
        team_records[away]["games"] += 1

    games["home_win_pct"] = home_win_pct
    games["away_win_pct"] = away_win_pct
    games["win_pct_diff"] = games["home_win_pct"] - games["away_win_pct"]

    return games
    
def add_recent_form(games):
    print("Adding recent form features...")

    games = games.sort_values("date")

    team_history = {}

    home_recent = []
    away_recent = []

    for _, row in games.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        if home not in team_history:
            team_history[home] = []
        if away not in team_history:
            team_history[away] = []

        def recent_win_pct(history):
            if len(history) == 0:
                return 0.5
            return sum(history[-10:]) / min(len(history), 10)
        
        home_recent.append(recent_win_pct(team_history[home]))
        away_recent.append(recent_win_pct(team_history[away]))

        team_history[home].append(row["home_win"])
        team_history[away].append(1 - row["home_win"])
    
    games["home_recent_win_pct"] = home_recent
    games["away_recent_win_pct"] = away_recent
    games["recent_win_pct_diff"] = games["home_recent_win_pct"] - games["away_recent_win_pct"]
    
    return games

def add_rolling_net_rating(df, window=10):

    df = df.copy()
    df = df.sort_values("date")

    # -----------------------------
    # STEP 1 — Build unified team table
    # -----------------------------
    home_df = df[[
        "date",
        "home_team",
        "home_net_rating"
    ]].rename(columns={
        "home_team": "team",
        "home_net_rating": "net_rating"
    })

    away_df = df[[
        "date",
        "away_team",
        "away_net_rating"
    ]].rename(columns={
        "away_team": "team",
        "away_net_rating": "net_rating"
    })

    team_games = pd.concat([home_df, away_df])
    team_games = team_games.sort_values(["team", "date"])

    # -----------------------------
    # STEP 2 — Rolling average (NO LEAKAGE)
    # -----------------------------
    team_games["rolling_net"] = (
        team_games.groupby("team")["net_rating"]
        .rolling(window, min_periods=1)
        .mean()
        .shift(1)   # 🔥 CRITICAL
        .reset_index(level=0, drop=True)
    )

    # -----------------------------
    # STEP 3 — Merge back to main df
    # -----------------------------
    df = df.merge(
        team_games[["team", "date", "rolling_net"]],
        left_on=["home_team", "date"],
        right_on=["team", "date"],
        how="left"
    ).rename(columns={"rolling_net": "home_rolling_net"}).drop(columns=["team"])

    df = df.merge(
        team_games[["team", "date", "rolling_net"]],
        left_on=["away_team", "date"],
        right_on=["team", "date"],
        how="left"
    ).rename(columns={"rolling_net": "away_rolling_net"}).drop(columns=["team"])

    # -----------------------------
    # STEP 4 — Final feature
    # -----------------------------
    df["rolling_net_diff"] = df["home_rolling_net"] - df["away_rolling_net"]

    # Fill early-game NaNs
    df["rolling_net_diff"] = df["rolling_net_diff"].fillna(0)

    return df


def update_elo(rating_a, rating_b, result, k=20):
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    rating_a_new = rating_a + k * (result - expected_a)
    rating_b_new = rating_b + k * ((1 - result) - (1 - expected_a))
    return rating_a_new, rating_b_new


def add_elo_features(df):

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "GAME_ID"], kind="mergesort")

    elo = {}
    K = 20

    home_elo_list = []
    away_elo_list = []

    for _, row in df.iterrows():

        home = row["home_team"]
        away = row["away_team"]

        if home not in elo:
            elo[home] = 1500
        if away not in elo:
            elo[away] = 1500

        ra = elo[home]
        rb = elo[away]

        # ✅ STORE BEFORE UPDATE
        home_elo_list.append(ra)
        away_elo_list.append(rb)

        # expected probability
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))

        result = row["home_win"]

        # update Elo
        elo[home] = ra + K * (result - ea)
        elo[away] = rb + K * ((1 - result) - (1 - ea))

    df["home_elo"] = home_elo_list
    df["away_elo"] = away_elo_list
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    return df

def add_rolling_off_def(df, window=10):

    df = df.copy()
    df = df.sort_values("date")

    # -----------------------------
    # STEP 1 — Team-level table
    # -----------------------------
    home_df = df[["date", "home_team", "home_off_rating", "away_off_rating"]].copy()
    home_df.columns = ["date", "team", "off", "opp_off"]

    away_df = df[["date", "away_team", "away_off_rating", "home_off_rating"]].copy()
    away_df.columns = ["date", "team", "off", "opp_off"]

    team_games = pd.concat([home_df, away_df])
    team_games = team_games.sort_values(["team", "date"])

    # -----------------------------
    # STEP 2 — Rolling offense
    # -----------------------------
    team_games["rolling_off"] = (
        team_games.groupby("team")["off"]
        .rolling(window, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # -----------------------------
    # STEP 3 — Rolling defense
    # -----------------------------
    team_games["rolling_def"] = (
        team_games.groupby("team")["opp_off"]
        .rolling(window, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # -----------------------------
    # STEP 4 — Merge back
    # -----------------------------
    df = df.merge(
        team_games[["team", "date", "rolling_off", "rolling_def"]],
        left_on=["home_team", "date"],
        right_on=["team", "date"],
        how="left"
    ).rename(columns={
        "rolling_off": "home_rolling_off",
        "rolling_def": "home_rolling_def"
    }).drop(columns=["team"])

    df = df.merge(
        team_games[["team", "date", "rolling_off", "rolling_def"]],
        left_on=["away_team", "date"],
        right_on=["team", "date"],
        how="left"
    ).rename(columns={
        "rolling_off": "away_rolling_off",
        "rolling_def": "away_rolling_def"
    }).drop(columns=["team"])

    # -----------------------------
    # STEP 5 — Final features
    # -----------------------------
    df["rolling_off_diff"] = df["home_rolling_off"] - df["away_rolling_off"]
    df["rolling_def_diff"] = df["home_rolling_def"] - df["away_rolling_def"]

    df[["rolling_off_diff", "rolling_def_diff"]] = df[[
        "rolling_off_diff", "rolling_def_diff"
    ]].fillna(0)

    df["home_off_vs_away_def"] = df["home_rolling_off"] - df["away_rolling_def"]
    df["away_off_vs_home_def"] = df["away_rolling_off"] - df["home_rolling_def"]

    df["matchup_diff"] = df["home_off_vs_away_def"] - df["away_off_vs_home_def"]
    return df

def add_rolling_true_shooting(df, window=10):

    df = df.copy()
    df = df.sort_values("date")

    # -----------------------------
    # STEP 1 — True Shooting (per game)
    # -----------------------------
    df["home_ts"] = df["home_points"] / (
        2 * (df["FGA_home"] + 0.44 * df["FTA_home"])
    )

    df["away_ts"] = df["away_points"] / (
        2 * (df["FGA_away"] + 0.44 * df["FTA_away"])
    )

    # -----------------------------
    # STEP 2 — Build unified team table
    # -----------------------------
    home_df = df[["date", "home_team", "home_ts"]].rename(columns={
        "home_team": "team",
        "home_ts": "ts"
    })

    away_df = df[["date", "away_team", "away_ts"]].rename(columns={
        "away_team": "team",
        "away_ts": "ts"
    })

    team_games = pd.concat([home_df, away_df])
    team_games = team_games.sort_values(["team", "date"])

    # -----------------------------
    # STEP 3 — Rolling TS (NO LEAKAGE)
    # -----------------------------
    team_games["rolling_ts"] = (
        team_games.groupby("team")["ts"]
        .rolling(window, min_periods=1)
        .mean()
        .shift(1)   # 🔥 CRITICAL
        .reset_index(level=0, drop=True)
    )

    # -----------------------------
    # STEP 4 — Merge back to main df
    # -----------------------------
    df = df.merge(
        team_games[["team", "date", "rolling_ts"]],
        left_on=["home_team", "date"],
        right_on=["team", "date"],
        how="left"
    ).rename(columns={"rolling_ts": "home_rolling_ts"}).drop(columns=["team"])

    df = df.merge(
        team_games[["team", "date", "rolling_ts"]],
        left_on=["away_team", "date"],
        right_on=["team", "date"],
        how="left"
    ).rename(columns={"rolling_ts": "away_rolling_ts"}).drop(columns=["team"])

    # -----------------------------
    # STEP 5 — Final feature
    # -----------------------------
    df["rolling_ts_diff"] = df["home_rolling_ts"] - df["away_rolling_ts"]

    # Handle early games
    df["rolling_ts_diff"] = df["rolling_ts_diff"].fillna(0)

    return df

def add_rolling_possessions(df, window=10):

    df = df.copy()
    df = df.sort_values("date")

    home_df = df[["date", "home_team", "home_possessions"]].rename(
        columns={"home_team": "team", "home_possessions": "pos"}
    )

    away_df = df[["date", "away_team", "away_possessions"]].rename(
        columns={"away_team": "team", "away_possessions": "pos"}
    )

    team_games = pd.concat([home_df, away_df]).sort_values(["team", "date"])

    team_games["rolling_pos"] = (
        team_games.groupby("team")["pos"]
        .rolling(window, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    df = df.merge(
        team_games[["team", "date", "rolling_pos"]],
        left_on=["home_team", "date"],
        right_on=["team", "date"],
        how="left"
    ).rename(columns={"rolling_pos": "home_rolling_pos"}).drop(columns=["team"])

    df = df.merge(
        team_games[["team", "date", "rolling_pos"]],
        left_on=["away_team", "date"],
        right_on=["team", "date"],
        how="left"
    ).rename(columns={"rolling_pos": "away_rolling_pos"}).drop(columns=["team"])

    df["rolling_pos_diff"] = df["home_rolling_pos"] - df["away_rolling_pos"]
    df["rolling_pos_diff"] = df["rolling_pos_diff"].fillna(0)

    return df

def add_point_diff(df):
    df["point_diff"] = df["home_points"] - df["away_points"]
    return df

def build_features():
    games = pd.read_csv("data/processed_games.csv")
    games["date"] = pd.to_datetime(games["date"])
    games = add_team_win_pct(games)
    games = add_recent_form(games)
    games = add_rolling_net_rating(games)
    games = add_elo_features(games)
    games = add_rolling_off_def(games)
    games = add_rolling_true_shooting(games)
    games = add_rolling_possessions(games)
    games = add_point_diff(games)

    return games

if __name__ == "__main__":
    games = pd.read_csv("data/processed_games.csv")
    games["date"] = pd.to_datetime(games["date"])
    games = add_team_win_pct(games)
    games = add_recent_form(games)
    games = add_rolling_net_rating(games)
    games = add_elo_features(games)
    games = add_rolling_off_def(games)
    games = add_rolling_true_shooting(games)
    games = add_rolling_possessions(games)
    games = add_point_diff(games)
    games["home_advantage"] = 1
    games.to_csv("data/featured_games.csv", index=False)
    print("Featured data saved to data/featured_games.csv")