"""
Generate sample data for the Climate Monitoring Tool.
Run this once to create sample images and sensor CSV.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import random

os.makedirs("data", exist_ok=True)
os.makedirs("data/sample_images", exist_ok=True)

def create_sample_image(filename, dominant="vegetation"):
    """Create simple colored sample images that look like aerial views."""
    img = Image.new("RGB", (400, 300), color=(100, 100, 100))
    draw = ImageDraw.Draw(img)

    if dominant == "vegetation":
        # Green areas
        for _ in range(30):
            x, y = random.randint(0, 350), random.randint(0, 250)
            w, h = random.randint(30, 80), random.randint(30, 60)
            color = (random.randint(20, 80), random.randint(120, 200), random.randint(20, 80))
            draw.ellipse([x, y, x+w, y+h], fill=color)
        # Some brown soil
        for _ in range(8):
            x, y = random.randint(0, 350), random.randint(0, 250)
            draw.rectangle([x, y, x+40, y+30], fill=(139, 90, 43))

    elif dominant == "water":
        draw.rectangle([0, 0, 400, 300], fill=(30, 90, 180))
        for _ in range(15):
            x, y = random.randint(0, 350), random.randint(0, 250)
            draw.ellipse([x, y, x+50, y+40], fill=(50, 130, 220))

    elif dominant == "urban":
        # Gray buildings and roads
        draw.rectangle([0, 0, 400, 300], fill=(80, 80, 80))
        for _ in range(12):
            x, y = random.randint(0, 320), random.randint(0, 220)
            draw.rectangle([x, y, x+60, y+50], fill=(120, 120, 130))
        # Roads
        draw.rectangle([0, 140, 400, 170], fill=(50, 50, 50))
        draw.rectangle([180, 0, 210, 300], fill=(50, 50, 50))

    elif dominant == "soil":
        draw.rectangle([0, 0, 400, 300], fill=(160, 120, 70))
        for _ in range(20):
            x, y = random.randint(0, 380), random.randint(0, 280)
            draw.ellipse([x, y, x+20, y+15], fill=(140, 100, 50))

    elif dominant == "mixed":
        # Mix of everything
        draw.rectangle([0, 0, 200, 150], fill=(40, 150, 50))      # vegetation
        draw.rectangle([200, 0, 400, 150], fill=(40, 100, 180))    # water
        draw.rectangle([0, 150, 200, 300], fill=(150, 110, 60))    # soil
        draw.rectangle([200, 150, 400, 300], fill=(100, 100, 110)) # urban

    img.save(f"data/sample_images/{filename}")
    print(f"Created: {filename}")


def create_sensor_data():
    """Create realistic-looking sensor CSV data."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2025-06-01", periods=n, freq="h")

    data = {
        "timestamp": dates,
        "temperature_c": np.round(28 + 6 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 1.5, n), 1),
        "humidity_pct": np.round(55 + 15 * np.sin(np.linspace(0, 3*np.pi, n)) + np.random.normal(0, 4, n), 1),
        "pressure_hpa": np.round(1010 + np.random.normal(0, 3, n), 1),
        "air_quality_aqi": np.clip(np.round(45 + 30 * np.sin(np.linspace(0, 2*np.pi, n)) + np.random.normal(0, 10, n), 0), 10, 200).astype(int),
        "light_lux": np.round(np.clip(8000 + 6000 * np.sin(np.linspace(0, 2*np.pi, n)) + np.random.normal(0, 500, n), 0, 12000), 0).astype(int),
        "latitude": 24.1985 + np.random.normal(0, 0.008, n),
        "longitude": 55.6810 + np.random.normal(0, 0.008, n),
    }

    df = pd.DataFrame(data)
    df.to_csv("data/sensor_data.csv", index=False)
    print("Created: data/sensor_data.csv")


def create_thermal_data():
    """Create simulated surface temperature grid for heat island."""
    np.random.seed(7)
    # 50x50 grid
    base = np.linspace(32, 42, 50)
    temp_grid = np.outer(base, np.ones(50)) + np.random.normal(0, 1.5, (50, 50))
    # Add a hot spot (urban center)
    temp_grid[18:32, 18:32] += 6
    # Cooler park area
    temp_grid[5:15, 30:45] -= 4
    np.save("data/thermal_grid.npy", temp_grid)
    print("Created: data/thermal_grid.npy")


if __name__ == "__main__":
    create_sample_image("vegetation_1.jpg", "vegetation")
    create_sample_image("vegetation_2.jpg", "vegetation")
    create_sample_image("water_1.jpg", "water")
    create_sample_image("urban_1.jpg", "urban")
    create_sample_image("soil_1.jpg", "soil")
    create_sample_image("mixed_1.jpg", "mixed")
    create_sensor_data()
    create_thermal_data()
    print("\nAll sample data generated successfully!")
