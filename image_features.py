# File: scripts/image_features.py

"""
image_features.py

Extracts CNN embeddings from food photos using MobileNetV2 (TensorFlow)
and saves them for modeling.
"""

import os
import joblib
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

def get_image_paths_and_labels(image_dir):
    image_paths, labels = [], []
    for cls in sorted(os.listdir(image_dir)):
        cls_dir = os.path.join(image_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(cls_dir, fname))
                labels.append(cls)
    return image_paths, labels

def extract_embeddings(image_paths, model, target_size=(224, 224)):
    batch = []
    for path in tqdm(image_paths, desc="Loading images"):
        img = load_img(path, target_size=target_size)
        arr = img_to_array(img)
        batch.append(preprocess_input(arr))
    batch = np.stack(batch)
    embeddings = model.predict(batch, verbose=1)
    return embeddings

def main():
    # Project‐root paths
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    image_dir = os.path.join(root, "data", "images")
    save_path = os.path.join(root, "models", "image_features.pkl")

    # 1) Load pretrained MobileNetV2
    base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

    # 2) Gather images
    image_paths, labels = get_image_paths_and_labels(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    print(f"Found {len(image_paths)} images across {len(set(labels))} classes.")

    # 3) Extract and save embeddings
    embeddings = extract_embeddings(image_paths, base_model)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({"embeddings": embeddings, "labels": labels, "paths": image_paths}, save_path)
    print(f"Saved image features to {save_path}")

if __name__ == "__main__":
    main()
