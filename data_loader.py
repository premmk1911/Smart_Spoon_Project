# File: scripts/data_loader.py

"""
data_loader.py

Loads the cleaned survey dataset from the project’s data/ folder,
simulates dummy stimulation targets for prototyping,
and prepares train/test splits using absolute paths so it works
regardless of your current working directory.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_survey_data(path=None):
    """
    Load the cleaned survey CSV and return a DataFrame.
    If no path is provided, computes the absolute path to data/cleaned_dataset.csv
    relative to the project root.
    """
    # Determine project root (one level up from scripts folder)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_path = path if path else os.path.join(root_dir, "data", "cleaned_dataset.csv")

    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Columns:", df.columns.tolist())
    return df


def simulate_targets(df):
    """
    Add dummy stimulation parameters (amplitude [mA], frequency [Hz], pulse_width [µs])
    so we can prototype the pipeline end-to-end.
    """
    np.random.seed(42)
    df["amplitude"]   = np.random.uniform(0.5, 5.0,  size=len(df))   # 0.5–5 mA
    df["frequency"]   = np.random.uniform(10, 100,  size=len(df))    # 10–100 Hz
    df["pulse_width"] = np.random.uniform(100, 500, size=len(df))    # 100–500 µs
    print("Simulated targets: ['amplitude', 'frequency', 'pulse_width']")
    return df, ["amplitude", "frequency", "pulse_width"]


def split_data(df, target_cols, test_size=0.2, random_state=42):
    """
    Split DataFrame into training and test sets.
    """
    X = df.drop(columns=target_cols)
    y = df[target_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # 1) Load
    df = load_survey_data()

    # 2) Simulate targets (for prototyping)
    df, targets = simulate_targets(df)

    # 3) Split
    X_train, X_test, y_train, y_test = split_data(df, targets)

    # 4) Quick inspect
    print("\nSample X_train columns:", X_train.columns.tolist()[:5], "…")
    print("Sample y_train head:\n", y_train.head())
