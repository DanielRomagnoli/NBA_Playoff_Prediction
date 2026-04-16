import pandas as pd

def process_games(df):
    print("Processing data...")

    df["is_home"] = df["MATCHUP"].str.contains("vs")

    home_df = df[df["is_home"]].copy()
    away_df = df[~df["is_home"]].copy()

    home_df = home_df.rename(columns={
        "TEAM_ABBREVIATION": "home_team",
        "PTS": "home_points",
        "WL": "home_wl",
        "FGA": "FGA_home",
        "FTA": "FTA_home",
        "OREB": "OREB_home",
        "TOV": "TOV_home"
    })

    away_df = away_df.rename(columns={
        "TEAM_ABBREVIATION": "away_team",
        "PTS": "away_points",
        "WL": "away_wl",
        "FGA": "FGA_away",
        "FTA": "FTA_away",
        "OREB": "OREB_away",
        "TOV": "TOV_away"
    })

    # Merge home and away dataframes
    merged_df = pd.merge(home_df, away_df, on="GAME_ID", suffixes=("_home", "_away"))

    merged_df["home_possessions"] = merged_df["FGA_home"] + 0.44 * merged_df["FTA_home"] - merged_df["OREB_home"] + merged_df["TOV_home"]

    merged_df["away_possessions"] = merged_df["FGA_away"] + 0.44 * merged_df["FTA_away"] - merged_df["OREB_away"] + merged_df["TOV_away"]

    merged_df["home_off_rating"] = merged_df["home_points"] / merged_df["home_possessions"] * 100
    merged_df["away_off_rating"] = merged_df["away_points"] / merged_df["away_possessions"] * 100
    merged_df["home_net_rating"] = merged_df["home_off_rating"] - merged_df["away_off_rating"]
    merged_df["away_net_rating"] = -merged_df["home_net_rating"]

    games = merged_df[[
        "GAME_ID",
        "GAME_DATE_home",
        "SEASON_home",
        "home_team",
        "home_points",
        "home_wl",
        "home_possessions",
        "home_off_rating",
        "home_net_rating",
        "away_team",
        "away_points",
        "away_wl",
        "away_possessions",
        "away_off_rating",
        "away_net_rating",
        "FGA_home",
        "FTA_home",
        "FGA_away",
        "FTA_away"
    ]].copy()

    games = games.rename(columns={"GAME_DATE_home": "date"})

    games["home_win"] = (games["home_wl"] == "W").astype(int)

    return games

if __name__ == "__main__":
    df = pd.read_csv("data/raw_games.csv")
    games = process_games(df)
    games.to_csv("data/processed_games.csv", index=False)
    print("Processed data saved to data/processed_games.csv")