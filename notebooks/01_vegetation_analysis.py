"""
Part 1 – Vegetation Analysis (Standalone Script)
Task 3 from the internship document.

How to run:
    python notebooks/01_vegetation_analysis.py
"""

import cv2
import numpy as np
from PIL import Image
import os

def calculate_green_ratio(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Simple green detection
    green_mask = (img_rgb[:, :, 1] > img_rgb[:, :, 0]) & \
                 (img_rgb[:, :, 1] > img_rgb[:, :, 2]) & \
                 (img_rgb[:, :, 1] > 60)

    ratio = (np.sum(green_mask) / green_mask.size) * 100

    if ratio >= 40:
        level = "High Vegetation"
    elif ratio >= 15:
        level = "Medium Vegetation"
    else:
        level = "Low Vegetation"

    return ratio, level


if __name__ == "__main__":
    sample_dir = "data/sample_images"
    print("=== Vegetation Analysis Results ===\n")

    for filename in os.listdir(sample_dir):
        if filename.endswith((".jpg", ".png")):
            path = os.path.join(sample_dir, filename)
            ratio, level = calculate_green_ratio(path)
            print(f"{filename:25s} → {ratio:5.1f}% green  |  {level}")
