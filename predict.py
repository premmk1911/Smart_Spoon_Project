# File: scripts/predict.py

import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ─── Setup ──────────────────────────────────────────────────────────────────────

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# 1) Survey pipeline + survey-only regressor
SURVEY_PIPE  = joblib.load(os.path.join(ROOT, "models", "survey_pipeline.pkl"))
SURVEY_MODEL = joblib.load(os.path.join(ROOT, "models", "survey_only_model.pkl"))

# 2) Dish classifier
DISH_CLF = load_model(os.path.join(ROOT, "models", "dish_classifier.h5"))

# 3) Rule-based dish adjustments
DISH_ADJUSTMENTS = {
    "dal":     np.array([ 0.5,  5.0,  20.0]),
    "sambar":  np.array([-0.2, 10.0, -15.0]),
    "biryani": np.array([ 0.0,  0.0,   0.0]),
    "curry":   np.array([ 0.3,  7.0,   5.0]),
    # add any others...
}


def predict(survey_answers: dict, image_path: str) -> dict:
    # 1) Survey branch
    df = pd.DataFrame([survey_answers])
    X_s = SURVEY_PIPE.transform(df)
    y_s = SURVEY_MODEL.predict(X_s)[0]

    # 2) Image branch
    img = load_img(image_path, target_size=(224, 224))
    arr = img_to_array(img)
    arr = preprocess_input(arr)
    probs = DISH_CLF.predict(np.expand_dims(arr, 0))[0]
    dish_idx = int(np.argmax(probs))
    dish = list(DISH_ADJUSTMENTS.keys())[dish_idx]

    # 3) Blend signals
    adj   = DISH_ADJUSTMENTS.get(dish, np.zeros(3))
    final = 0.8 * y_s + 0.2 * adj

    return {
        "dish":        dish,
        "amplitude":   float(final[0]),
        "frequency":   float(final[1]),
        "pulse_width": float(final[2]),
    }


if __name__ == "__main__":
    import sys, json

    if len(sys.argv) != 3:
        print("Usage: python scripts/predict.py <survey.json> <image_path>")
        sys.exit(1)

    survey_file = sys.argv[1]
    image_file  = sys.argv[2]

    # Load the demo survey JSON
    try:
        with open(survey_file, 'r', encoding='utf-8-sig') as f:
            survey_dict = json.load(f)
    except Exception as e:
        print(f"Error reading survey file: {e}")
        sys.exit(1)


    # Run prediction
    result = predict(survey_dict, image_file)

    # Print results so you can see them
    print("Predicted stimulation settings:")
    print(json.dumps(result, indent=2))
