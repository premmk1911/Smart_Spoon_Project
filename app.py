# File: app/app.py

import os
import sys

# ─── Ensure project root is on sys.path ────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.utils import secure_filename

from scripts.predict import predict   # now resolves correctly

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "supersecret")  # for flash messages

# Where uploads go
UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ─── Mapping dicts ─────────────────────────────────────────────────────────────

gender_map       = {"Male": 0, "Female": 1, "Other": 2}
yesno_map        = {"Yes": 1, "No": 0}
cond_map         = {"Hypertension": 1, "Kidney Disease": 2, "Other": 3}
freq_map         = {"Daily": 0, "Weekly": 1, "Monthly": 2, "Rarely": 3}
sat_map          = {"Yes": 1, "No": 0, "Sometimes": 2}
salt_cond_map    = {"Always": 2, "Sometimes": 1, "Never": 0}
importance_map   = {"Very Important": 2, "Moderately Important": 1, "Not Important": 0}
purchase_map     = {"Yes": 1, "No": 0, "Maybe": 2}
salt_opinion_map = {"Perfect": 0, "Too Salty": 1, "Too Bland": 2}

salt_usage_map = {
    "No Salt": 0,
    "¼ tsp":     1,
    "½ tsp":     2,
    "1 tsp":     3,
    "More than 1 tsp": 4
}

salt_fields = [
    "salt_usage_dal_gojju_palya",
    "salt_usage_sambar_rasam_curd",
    "salt_usage_biryani_pulao_rice",
    "salt_usage_curry",
    "salt_usage_snacks",
    "salt_usage_roti_paratha",
    "salt_usage_pickles_papad"
]

CLASS_LIST = ["dal", "sambar", "biryani", "curry", "snacks", "roti", "pickles"]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 1) Save uploaded image
        file = request.files.get("dish_image")
        if not file or file.filename == "":
            flash("Please upload a dish image.", "danger")
            return redirect(request.url)

        fname = secure_filename(file.filename)
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        file.save(img_path)

        # 2) Pull raw form data
        raw = request.form

        # 3) Map survey answers to numeric
        survey_answers = {
            "age": int(raw["age"]),
            "gender": gender_map[raw["gender"]],
            "low_sodium_diet": yesno_map[raw["low_sodium_diet"]],
            "diet_condition": cond_map.get(raw.get("diet_condition", ""), 0),
            "dining_frequency": freq_map[raw["dining_frequency"]],
            "low_sodium_satisfaction": sat_map[raw["low_sodium_satisfaction"]],
            "add_salt_condiments": salt_cond_map[raw["add_salt_condiments"]],
            "taste_enhancement_tech_aware": yesno_map[raw["taste_enhancement_tech_aware"]],
            "interest_in_device": yesno_map[raw["interest_in_device"]],
            "importance_of_taste_enhancement": importance_map[raw["importance_of_taste_enhancement"]],
            "purchase_consideration": purchase_map[raw["purchase_consideration"]],
            "concerns_on_technology": int(raw["concerns_on_technology"]),
            "salt_opinion": salt_opinion_map[raw["salt_opinion"]],
        }

        # Count checked expected features
        cnt = 0
        for feat in ["adjustable", "portable", "app", "tracking"]:
            if raw.get("expected_device_features") == feat:
                cnt += 1
        survey_answers["expected_device_features"] = cnt

        # Section 6: salt usage fields
        for fld in salt_fields:
            survey_answers[fld] = salt_usage_map[raw[fld]]

        # 4) Predict via scripts/predict.py
        result = predict(survey_answers, img_path)
        dish_detected = result["dish"]

        # 5) Manual override
        manual = raw.get("manual_override", "")
        dish_used = manual if manual else dish_detected

        # 6) Render results
        return render_template(
            "result.html",
            image_url=url_for("static", filename=f"uploads/{fname}"),
            detected=dish_detected.capitalize(),
            used=dish_used.capitalize(),
            amplitude=result["amplitude"],
            frequency=result["frequency"],
            pulse_width=result["pulse_width"],
            class_list=CLASS_LIST
        )

    # GET → show form
    return render_template("index.html", class_list=CLASS_LIST)


if __name__ == "__main__":
    app.run(debug=True)
