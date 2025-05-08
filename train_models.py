# File: scripts/train_models.py

"""
train_models.py

Trains a survey‐only model that predicts (amplitude, frequency, pulse_width)
solely from the processed survey features.
"""

import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from data_loader import load_survey_data, simulate_targets, split_data

def main():
    # 1) Load & split survey data
    df = load_survey_data()
    df, target_cols = simulate_targets(df)
    X_train, X_test, y_train, y_test = split_data(df, target_cols)

    # 2) Load & apply the survey pipeline
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    survey_pipe = joblib.load(os.path.join(root, "models", "survey_pipeline.pkl"))
    X_train_s = survey_pipe.transform(X_train)
    X_test_s  = survey_pipe.transform(X_test)

    print("Survey-only shapes:", X_train_s.shape, X_test_s.shape)

    # 3) Train a regressor on survey data alone
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_s, y_train)

    # 4) Evaluate & save
    print("Test R² (survey-only):", model.score(X_test_s, y_test))
    os.makedirs(os.path.join(root, "models"), exist_ok=True)
    joblib.dump(model, os.path.join(root, "models", "survey_only_model.pkl"))
    print("Saved survey-only model to models/survey_only_model.pkl")

if __name__ == "__main__":
    main()
