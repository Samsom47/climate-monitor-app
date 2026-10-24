"""
Fake drone simulation for testing live telemetry.
Writes sensor rows to live_data/live_sensors.csv and
updates live_data/live_frame.jpg every interval seconds.

Usage (from project root):
    python tools/fake_drone_sim.py
    python tools/fake_drone_sim.py --interval 2 --rows 60
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_DIR = os.path.join(ROOT, "live_data")
CSV_PATH = os.path.join(LIVE_DIR, "live_sensors.csv")
FRAME_PATH = os.path.join(LIVE_DIR, "live_frame.jpg")

COLUMNS = [
    "timestamp",
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "air_quality_aqi",
    "light_lux",
    "latitude",
    "longitude",
    "speed_mps",
]


def ensure_live_dir():
    os.makedirs(LIVE_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        pd.DataFrame(columns=COLUMNS).to_csv(CSV_PATH, index=False)


def append_sensor_row(i: int, lat: float, lon: float):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": now,
        "temperature_c": round(28 + 6 * np.sin(i / 8) + np.random.normal(0, 0.4), 1),
        "humidity_pct": round(50 + 12 * np.sin(i / 10) + np.random.normal(0, 1.5), 1),
        "pressure_hpa": round(1010 + np.random.normal(0, 1.2), 1),
        "air_quality_aqi": int(np.clip(40 + 25 * np.sin(i / 12) + np.random.normal(0, 5), 15, 160)),
        "light_lux": int(np.clip(5000 + 4000 * np.sin(i / 9) + np.random.normal(0, 200), 0, 12000)),
        "latitude": round(lat, 5),
        "longitude": round(lon, 5),
        "speed_mps": round(float(np.clip(5 + 2.5 * np.sin(i / 7) + np.random.normal(0, 0.4), 0.5, 12)), 1),
    }
    df = pd.DataFrame([row])
    header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
    df.to_csv(CSV_PATH, mode="a", header=header, index=False)
    return row


def write_frame(i: int, row: dict):
    """Simple synthetic camera frame with telemetry overlay."""
    img = Image.new("RGB", (640, 360), (25, 35, 50))
    draw = ImageDraw.Draw(img)
    # sky / ground split
    draw.rectangle([0, 0, 640, 160], fill=(40, 70, 110))
    draw.rectangle([0, 160, 640, 360], fill=(50, 90, 45))
    # fake "drone path" marker
    x = 50 + (i * 7) % 540
    y = 180 + int(20 * np.sin(i / 5))
    draw.ellipse([x, y, x + 24, y + 24], fill=(95, 211, 141), outline=(230, 240, 255))
    # HUD text
    lines = [
        "FAKE DRONE CAM FEED",
        f"t = {row['timestamp']}",
        f"Temp {row['temperature_c']} C   Hum {row['humidity_pct']}%",
        f"AQI {row['air_quality_aqi']}   Speed {row.get('speed_mps', 0)} m/s",
        f"GPS {row['latitude']}, {row['longitude']}",
        f"Frame #{i}",
    ]
    yy = 12
    for line in lines:
        draw.text((14, yy), line, fill=(220, 230, 240))
        yy += 18
    img.save(FRAME_PATH, quality=85)


def main():
    parser = argparse.ArgumentParser(description="Fake drone live data writer")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between updates")
    parser.add_argument("--rows", type=int, default=0, help="Stop after N rows (0 = run forever)")
    parser.add_argument("--reset", action="store_true", help="Clear existing live CSV before start")
    args = parser.parse_args()

    ensure_live_dir()
    if args.reset and os.path.exists(CSV_PATH):
        pd.DataFrame(columns=COLUMNS).to_csv(CSV_PATH, index=False)
        print("Reset live_sensors.csv")

    lat, lon = 24.1985, 55.6810  # UAE University, Al Ain
    print(f"Writing live data to:\n  {CSV_PATH}\n  {FRAME_PATH}")
    print("Open the website Sensor Dashboard → Live from simulation")
    print("Press Ctrl+C to stop.\n")

    i = 0
    try:
        while True:
            lat += np.random.normal(0, 0.00015)
            lon += np.random.normal(0, 0.00015)
            row = append_sensor_row(i, lat, lon)
            write_frame(i, row)
            print(f"[{row['timestamp']}] temp={row['temperature_c']} AQI={row['air_quality_aqi']} frame ok")
            i += 1
            if args.rows and i >= args.rows:
                print("Done.")
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
