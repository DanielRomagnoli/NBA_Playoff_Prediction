# NBA_Playoff_Prediction

Overview

Built a machine learning model to predict NBA game outcomes and simulate full playoff brackets using Monte Carlo methods.

Features
Logistic regression model (log loss ~0.60)
Custom Elo rating system
Rolling team performance metrics
Playoff bracket simulator
Monte Carlo championship probabilities
Realistic series modeling (7-game dynamics)
Tech Stack
Python (Pandas, NumPy, Scikit-learn)
XGBoost
Monte Carlo simulation
Model Highlights
Engineered features including:
Elo difference
Rolling offensive/defensive ratings
Matchup-based features
Achieved calibrated probabilities for realistic simulations
Simulation Engine
Series-level variance modeling
Seed-based adjustments
Realistic playoff outcome distributions
Results
Accurate game-level predictions
Realistic playoff series distributions:
4–0: ~22%
4–1: ~30%
4–2: ~25%
4–3: ~21%


How To Run:
streamlit run app.py