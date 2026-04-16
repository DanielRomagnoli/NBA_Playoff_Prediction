from simulate import  build_initial_state, predict_game, simulate_series, simulate_bracket_stable, build_series_state
import pandas as pd
import joblib

df = pd.read_csv("data/featured_games.csv")
state = build_initial_state(df)
series_state = build_series_state(df)
model = joblib.load("models/model.pkl")

def test_series_distribution(state):

    counts = {"4-0":0, "4-1":0, "4-2":0, "4-3":0}

    for _ in range(1000):
        winner, wA, wB = simulate_series(state, "OKC", "SAS")

        score = f"{max(wA,wB)}-{min(wA,wB)}"
        counts[score] += 1

    for k in counts:
        counts[k] /= 1000

    print(counts)



print(test_series_distribution(state))
