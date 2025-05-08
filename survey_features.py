# File: scripts/survey_features.py

"""
survey_features.py

Builds and applies a feature‐processing pipeline for your survey data:
 - OneHotEncodes categorical answers
 - StandardScales numerical answers
 - Saves the fitted pipeline to models/survey_pipeline.pkl
"""

import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import loader and split functions from our scripts folder
from data_loader import load_survey_data, simulate_targets, split_data

# Define which columns are categorical vs numerical
CATEGORICAL_FEATURES = [
    "gender",
    "low_sodium_diet",
    "diet_condition",
    "low_sodium_satisfaction",
    "add_salt_condiments",
    "taste_enhancement_tech_aware",
    "interest_in_device",
    "importance_of_taste_enhancement",
    "expected_device_features",
    "purchase_consideration",
    "concerns_on_technology",
    "salt_opinion",
]

NUMERICAL_FEATURES = [
    "age",
    "dining_frequency",
    "salt_usage_dal_gojju_palya",
    "salt_usage_sambar_rasam_curd",
    "salt_usage_biryani_pulao_rice",
    "salt_usage_curry",
    "salt_usage_snacks",
    "salt_usage_roti_paratha",
    "salt_usage_pickles_papad",
]


def build_survey_pipeline():
    """
    Construct a ColumnTransformer that applies:
      - OneHotEncoder to categorical features
      - StandardScaler to numerical features
    """
    transformer = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERICAL_FEATURES),
        ],
        remainder="drop"
    )
    return transformer


def extract_and_save(X_train, X_test, save_path=None):
    """
    Fit the pipeline on X_train, transform both X_train and X_test,
    and save the fitted pipeline to disk.
    """
    # Determine project root to save under <root>/models/
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if save_path is None:
        save_path = os.path.join(root_dir, "models", "survey_pipeline.pkl")

    pipeline = build_survey_pipeline()
    X_train_trans = pipeline.fit_transform(X_train)
    X_test_trans  = pipeline.transform(X_test)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)

    print(f"Pipeline saved to {save_path}")
    print(f"Transformed X_train shape: {X_train_trans.shape}")
    print(f"Transformed X_test  shape: {X_test_trans.shape}")

    return X_train_trans, X_test_trans, pipeline


if __name__ == "__main__":
    # 1) Load raw survey data
    df = load_survey_data()

    # 2) Simulate targets (for prototyping)
    df, target_cols = simulate_targets(df)

    # 3) Split into train/test
    X_train, X_test, y_train, y_test = split_data(df, target_cols)

    # 4) Build pipeline, transform, and save
    extract_and_save(X_train, X_test)
