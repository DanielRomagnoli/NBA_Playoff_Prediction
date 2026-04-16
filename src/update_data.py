import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
import time

SEASONS = [
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26"
]

def fetch_games():
    all_games = []
    for season in SEASONS:
        print(f"Fetching data for season {season}...")
        gamefinder = leaguegamelog.LeagueGameLog(season=season)
        games = gamefinder.get_data_frames()[0]
        games["SEASON"] = season  # Add season column for clarity
        all_games.append(games)
        time.sleep(2) 
        gamefinder = leaguegamelog.LeagueGameLog(season=season, season_type_all_star='Playoffs')
        games = gamefinder.get_data_frames()[0]
        games["SEASON"] = season  # Add season column for clarity
        all_games.append(games)
        time.sleep(2)  # Sleep to avoid hitting API rate limits
        print(f"Data fetched for season {season}: {games.shape[0]} rows, {games.shape[1]} columns")

    df = pd.concat(all_games, ignore_index=True)
    print(f"Data fetched: {df.shape[0]} rows, {df.shape[1]} columns")

    return df

if __name__ == "__main__":
    df = fetch_games()
    print(df["GAME_ID"].value_counts().head())  # Check for duplicate game IDs
    df.to_csv("data/raw_games.csv", index=False)
    print("Data saved to data/raw_games.csv")
