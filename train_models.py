"""
Train simple models for the Climate Monitoring Tool.
Uses only PIL + numpy + scikit-learn (no OpenCV needed).
"""

import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

os.makedirs("models", exist_ok=True)

def extract_features(image_path):
    """Extract simple color features from an image using PIL + numpy."""
    img = Image.open(image_path).convert("RGB").resize((64, 64))
    arr = np.array(img).astype(np.float32)

    # Average colors
    avg_r = arr[:, :, 0].mean()
    avg_g = arr[:, :, 1].mean()
    avg_b = arr[:, :, 2].mean()

    # Std (simple texture)
    std_r = arr[:, :, 0].std()
    std_g = arr[:, :, 1].std()
    std_b = arr[:, :, 2].std()

    # Green ratio
    green_mask = (arr[:, :, 1] > arr[:, :, 0]) & (arr[:, :, 1] > arr[:, :, 2]) & (arr[:, :, 1] > 60)
    green_ratio = green_mask.sum() / green_mask.size

    # Blue ratio (for water)
    blue_mask = (arr[:, :, 2] > arr[:, :, 0]) & (arr[:, :, 2] > arr[:, :, 1]) & (arr[:, :, 2] > 80)
    blue_ratio = blue_mask.sum() / blue_mask.size

    # Simple "brightness"
    brightness = arr.mean()

    return [avg_r, avg_g, avg_b, std_r, std_g, std_b, green_ratio, blue_ratio, brightness]


def create_training_data():
    """Create labeled data from sample images + synthetic variations."""
    X = []
    y = []

    # 0: Vegetation, 1: Water, 2: Soil, 3: Urban, 4: Mixed
    samples = {
        "data/sample_images/vegetation_1.jpg": 0,
        "data/sample_images/vegetation_2.jpg": 0,
        "data/sample_images/water_1.jpg": 1,
        "data/sample_images/soil_1.jpg": 2,
        "data/sample_images/urban_1.jpg": 3,
        "data/sample_images/mixed_1.jpg": 4,
    }

    for path, label in samples.items():
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue

        # Original
        feats = extract_features(path)
        X.append(feats)
        y.append(label)

        # Create simple augmented versions (brightness / noise)
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32)

        for i in range(10):
            noisy = arr + np.random.normal(0, 10, arr.shape)
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            # slight brightness change
            factor = np.random.uniform(0.85, 1.15)
            noisy = np.clip(noisy * factor, 0, 255).astype(np.uint8)

            temp_path = f"data/temp_aug_{label}_{i}.jpg"
            Image.fromarray(noisy).save(temp_path)
            feats = extract_features(temp_path)
            X.append(feats)
            y.append(label)
            os.remove(temp_path)

    return np.array(X), np.array(y)


def train_classifier():
    print("Creating training data...")
    X, y = create_training_data()
    print(f"Total samples: {len(X)}")

    if len(X) < 10:
        print("Not enough data!")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("Training RandomForest classifier...")
    model = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy on test set: {acc*100:.1f}%")
    print(classification_report(
        y_test, preds,
        target_names=["Vegetation", "Water", "Soil", "Urban", "Mixed"]
    ))

    # Save model
    model_path = "models/landcover_classifier.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved → {model_path}")

    # Save label mapping
    label_map = {
        0: "Vegetation",
        1: "Water",
        2: "Sand / Soil",
        3: "Buildings / Roads",
        4: "Mixed / Other"
    }
    joblib.dump(label_map, "models/label_map.joblib")
    print("Label map saved.")

    feature_names = [
        "avg_r", "avg_g", "avg_b",
        "std_r", "std_g", "std_b",
        "green_ratio", "blue_ratio", "brightness"
    ]
    joblib.dump(feature_names, "models/feature_names.joblib")
    print("Feature names saved.")

    return model


if __name__ == "__main__":
    train_classifier()

    # Write a short readme
    with open("models/README_models.txt", "w") as f:
        f.write("""Models Folder
=============

1. landcover_classifier.joblib
   - Real trained RandomForest model
   - Classes: Vegetation, Water, Sand/Soil, Buildings/Roads, Mixed
   - Input: 9 simple color & texture features extracted from the image

2. label_map.joblib
   - Maps class numbers to human-readable names

3. feature_names.joblib
   - Names of the features the model expects

How it works:
- Image → extract color averages, std, green/blue ratios → model predicts class
- This is a real machine learning model (not just if-else rules)

Vegetation green-ratio tool still uses direct pixel calculation
because it is more accurate and easy for students to understand.
""")
    print("\nmodels/README_models.txt created.")
    print("✅ Models ready!")
