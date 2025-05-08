# File: scripts/evaluate_classifier.py

import os, joblib, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = os.path.join("models", "dish_classifier.h5")
CI_PATH    = os.path.join("models", "class_indices.pkl")
DATA_DIR   = os.path.join("data", "images")

clf = load_model(MODEL_PATH)
ci  = joblib.load(CI_PATH)
idx2cls = {v:k for k, v in ci.items()}

correct = total = 0
for cls_name, idx in ci.items():
    cls_folder = os.path.join(DATA_DIR, cls_name)
    for imgfile in os.listdir(cls_folder):
        if not imgfile.lower().endswith((".jpg","jpeg","png")):
            continue
        img = load_img(os.path.join(cls_folder, imgfile), target_size=(224,224))
        arr = preprocess_input(img_to_array(img)[None,...])
        probs = clf.predict(arr)[0]
        pred = idx2cls[int(np.argmax(probs))]
        total += 1
        if pred == cls_name:
            correct += 1

print(f"Validation accuracy: {correct}/{total} = {correct/total:.3f}")
