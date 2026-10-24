"""
Part 2 – Image Classification using the trained model
Task 2 from the internship document.

How to run:
    python notebooks/02_image_classification.py
"""

import os
import numpy as np
from PIL import Image
import joblib

def extract_features(image_path):
    img = Image.open(image_path).convert("RGB").resize((64, 64))
    arr = np.array(img).astype(np.float32)

    avg_r = arr[:, :, 0].mean()
    avg_g = arr[:, :, 1].mean()
    avg_b = arr[:, :, 2].mean()
    std_r = arr[:, :, 0].std()
    std_g = arr[:, :, 1].std()
    std_b = arr[:, :, 2].std()

    green_mask = (arr[:, :, 1] > arr[:, :, 0]) & (arr[:, :, 1] > arr[:, :, 2]) & (arr[:, :, 1] > 60)
    green_ratio = green_mask.sum() / green_mask.size

    blue_mask = (arr[:, :, 2] > arr[:, :, 0]) & (arr[:, :, 2] > arr[:, :, 1]) & (arr[:, :, 2] > 80)
    blue_ratio = blue_mask.sum() / blue_mask.size

    brightness = arr.mean()

    return np.array([[avg_r, avg_g, avg_b, std_r, std_g, std_b, green_ratio, blue_ratio, brightness]])


if __name__ == "__main__":
    model = joblib.load("models/landcover_classifier.joblib")
    label_map = joblib.load("models/label_map.joblib")

    sample_dir = "data/sample_images"
    print("=== Image Classification Results (Trained Model) ===\n")

    for filename in sorted(os.listdir(sample_dir)):
        if filename.endswith((".jpg", ".png")):
            path = os.path.join(sample_dir, filename)
            feats = extract_features(path)
            pred = model.predict(feats)[0]
            conf = model.predict_proba(feats)[0][pred]
            label = label_map[pred]
            print(f"{filename:25s} → {label:20s} (confidence: {conf*100:.0f}%)")
