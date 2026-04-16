from src.update_data import fetch_games
from src.data_process import process_games
from src.feature_engineering import build_features
from src.simulate import (
    build_initial_state,
    build_series_state,
    monte_carlo
)

import pandas as pd


def run():

    # 1. FETCH DATA
    raw = fetch_games()
    raw.to_csv("data/raw_games.csv", index=False)

    # 2. PROCESS DATA
    games = process_games(raw)
    games.to_csv("data/processed_games.csv", index=False)

    # 3. FEATURES
    features = build_features()
    features.to_csv("data/features_games.csv", index=False)

    # 4. BUILD STATE
    state = build_initial_state()

    # 5. BUILD SERIES STATE (THIS IS YOUR QUESTION)
    series_state = build_series_state(features)

    print("Current Series State:")
    print(series_state)

    # 6. RUN SIMULATION
    results = monte_carlo(state, series_state)

    print("\nChampionship Probabilities:")
    for team, prob in results.items():
        print(f"{team}: {prob:.2%}")


if __name__ == "__main__":
    run()