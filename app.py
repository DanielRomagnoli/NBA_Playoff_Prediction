import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.simulate import (
    build_initial_state,
    build_series_state,
    monte_carlo,
    simulate_bracket_stable
)

from collections import defaultdict

import plotly.express as px

EAST = ["DET","BOS","NYK","CLE","TOR","ATL","PHI","CHA"]
WEST = ["OKC","SAS","DEN","LAL","HOU","MIN","POR","PHX"]

def reorder_r1(series_list, conference):

    # Original order from simulation:
    # [1v8, 2v7, 3v6, 4v5]

    # Correct visual order:
    order = [0, 3, 2, 1]

    return [series_list[i] for i in order]

def plot_pie(results):

    df = pd.DataFrame({
        "Team": list(results.keys()),
        "Probability": list(results.values())
    })

    fig = px.pie(
        df,
        names="Team",
        values="Probability",
        title="Championship Probabilities"
    )

    st.plotly_chart(fig, use_container_width=True)


def render_espn_bracket(bracket):

    def box(s, x, y):
        return f"""
        <div class="box" style="left:{x}px; top:{y}px;">
            <div>{s['team_A']} ({s['wins_A']})</div>
            <div>{s['team_B']} ({s['wins_B']})</div>
            <div class="winner">🏆 {s['winner']}</div>
        </div>
        """

    r1 = bracket["Round 1"]
    r2 = bracket["Round 2"]
    r3 = bracket["Conference Finals"]
    finals = bracket["Finals"][0]

    east_r1 = reorder_r1(r1[:4], EAST)
    west_r1 = reorder_r1(r1[4:], WEST)
    east_r2 = r2[:2]
    west_r2 = r2[2:]
    east_cf = r3[0]
    west_cf = r3[1]

    # X positions (columns)
    x_r1_e, x_r2_e, x_cf_e = 50, 260, 470
    x_final = 620
    x_cf_w, x_r2_w, x_r1_w = 770, 980, 1190

    # BASE SPACING
    base_gap = 120
    start_y = 60

    # R1 positions
    y_r1 = [start_y + i * base_gap for i in range(4)]

    # R2 = midpoint of R1 pairs
    y_r2 = [
        (y_r1[0] + y_r1[1]) // 2,
        (y_r1[2] + y_r1[3]) // 2
    ]

    # CF = midpoint of R2
    y_cf = (y_r2[0] + y_r2[1]) // 2

    # Finals = center
    y_final = y_cf

    html = f"""
    <style>
    body {{
        background: #0b0f1a;
        font-family: Arial;
        color: white;
    }}

    .canvas {{
        position: relative;
        width: 1300px;
        height: 650px;
        margin: auto;
    }}

    .box {{
        position: absolute;
        width: 140px;
        padding: 6px;
        background: #111;
        border: 1px solid #aaa;
        border-radius: 6px;
        text-align: center;
        font-size: 12px;
    }}

    .winner {{
        color: #00ff99;
        font-weight: bold;
        margin-top: 3px;
    }}
    </style>

    <div class="canvas">

    <!-- EAST -->
    {box(east_r1[0], x_r1_e, y_r1[0])}
    {box(east_r1[1], x_r1_e, y_r1[1])}
    {box(east_r1[2], x_r1_e, y_r1[2])}
    {box(east_r1[3], x_r1_e, y_r1[3])}

    {box(east_r2[0], x_r2_e, y_r2[0])}
    {box(east_r2[1], x_r2_e, y_r2[1])}

    {box(east_cf, x_cf_e, y_cf)}

    <!-- FINALS -->
    {box(finals, x_final, y_final)}

    <!-- WEST -->
    {box(west_cf, x_cf_w, y_cf)}

    {box(west_r2[0], x_r2_w, y_r2[0])}
    {box(west_r2[1], x_r2_w, y_r2[1])}

    {box(west_r1[0], x_r1_w, y_r1[0])}
    {box(west_r1[1], x_r1_w, y_r1[1])}
    {box(west_r1[2], x_r1_w, y_r1[2])}
    {box(west_r1[3], x_r1_w, y_r1[3])}

    </div>
    """

    st.html(html)

def display_series_probabilities(series_results):

    st.subheader("📊 Series Win Probabilities")

    for (team_A, team_B), data in series_results.items():

        total = data["count"]
        pA = data["A_wins"] / total
        pB = data["B_wins"] / total

        st.write(f"{team_A} vs {team_B}")
        st.write(f"{team_A}: {pA:.2%} | {team_B}: {pB:.2%}")
        st.progress(int(max(pA, pB) * 100))

st.title("🏀 NBA Playoff Predictor")

# Load data
df = pd.read_csv("data/featured_games.csv")

# Build state
state = build_initial_state(df)
series_state = build_series_state(df)

st.subheader("Current Series State")

if series_state:
    for k, v in series_state.items():
        st.write(f"{k}: {v}")
else:
    st.write("Playoffs not started yet")

# Simulation control
n = st.slider("Simulations", 100, 2000, 500)

# -----------------------------
# RUN SIMULATION
# -----------------------------
if st.button("Run Simulation"):

    # ---- Single bracket ----
    bracket = simulate_bracket_stable(state, series_state, n)

    render_espn_bracket(bracket)

    # ---- Champion probabilities ----
    results = monte_carlo(state, series_state, n)

    st.subheader("🏆 Championship Probabilities")
    plot_pie(results)