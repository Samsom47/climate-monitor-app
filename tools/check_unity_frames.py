#!/usr/bin/env python
"""
Quick check: what does the land-cover model predict on Unity drone frames?

Run from the SpacePoint_Climate_Tool folder (the one that contains app.py and models/).

  # 1) While Unity is in Play mode: grab 40 frames (one every 2 s) and classify them
  python tools/check_unity_frames.py --capture 40 --interval 2

  # 2) Classify frames you already saved (default folder: data/unity_frames)
  python tools/check_unity_frames.py

  # 3) Any other folder of images
  python tools/check_unity_frames.py --folder path/to/images

  # 4) Optional: compare with your own labels (CSV with columns: file,true_label)
  python tools/check_unity_frames.py --truth data/unity_frames/truth.csv

Outputs (inside the frames folder):
  classification_report.csv   one row per frame
  contact_sheet.png           thumbnails with the predicted label, for a quick visual check

Tip: capture across a WHOLE mission (e.g. --capture 60 --interval 2 = 2 minutes) so the frames
include the ground, buildings, trees and both zones.

NOTE: extract_features() below must stay identical to extract_features_from_image() in app.py.
"""
import argparse
import io
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
LIVE_FRAME = ROOT / "live_data" / "live_frame.jpg"
DEFAULT_FOLDER = ROOT / "data" / "unity_frames"
MODEL_PATH = ROOT / "models" / "landcover_classifier.joblib"
LABEL_PATH = ROOT / "models" / "label_map.joblib"
IMAGE_EXT = {".jpg", ".jpeg", ".png"}


# ----------------------------------------------------------------- model (same logic as app.py)
def load_model():
    if MODEL_PATH.exists() and LABEL_PATH.exists():
        return joblib.load(MODEL_PATH), joblib.load(LABEL_PATH)
    return None, None


def extract_features(image):
    img = image.convert("RGB").resize((64, 64))
    arr = np.array(img).astype(np.float32)
    avg_r, avg_g, avg_b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
    std_r, std_g, std_b = arr[:, :, 0].std(), arr[:, :, 1].std(), arr[:, :, 2].std()
    green_mask = (arr[:, :, 1] > arr[:, :, 0]) & (arr[:, :, 1] > arr[:, :, 2]) & (arr[:, :, 1] > 60)
    blue_mask = (arr[:, :, 2] > arr[:, :, 0]) & (arr[:, :, 2] > arr[:, :, 1]) & (arr[:, :, 2] > 80)
    return np.array([[avg_r, avg_g, avg_b, std_r, std_g, std_b,
                      green_mask.sum() / green_mask.size, blue_mask.sum() / blue_mask.size, arr.mean()]])


def rule_based(image):
    img = np.array(image.convert("RGB").resize((100, 100)))
    r, g, b = img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()
    if g > r + 15 and g > b + 10:
        return "Vegetation", 0.75
    if b > r + 20 and b > g + 10:
        return "Water", 0.80
    if abs(r - g) < 20 and abs(g - b) < 20 and r < 120:
        return "Buildings / Roads", 0.65
    if r > 100 and g > 70 and b < 90:
        return "Sand / Soil", 0.70
    return "Mixed / Other", 0.55


def classify(image, model, label_map):
    """-> (label, confidence, runner_up_label, runner_up_confidence)"""
    if model is None:
        label, conf = rule_based(image)
        return label, conf, "", 0.0
    probs = model.predict_proba(extract_features(image))[0]
    names = [label_map.get(c, str(c)) for c in list(model.classes_)]
    order = np.argsort(probs)[::-1]
    best = int(order[0])
    second = int(order[1]) if len(order) > 1 else best
    return names[best], float(probs[best]), (names[second] if second != best else ""), float(probs[second] if second != best else 0.0)


def green_ratio_pct(image):
    arr = np.array(image.convert("RGB"))
    mask = (arr[:, :, 1] > arr[:, :, 0]) & (arr[:, :, 1] > arr[:, :, 2]) & (arr[:, :, 1] > 60)
    return 100.0 * mask.sum() / mask.size


# ----------------------------------------------------------------- capture
def capture_frames(n, interval, folder):
    """Copy a NEW live_frame.jpg into `folder` every time Unity overwrites it."""
    folder.mkdir(parents=True, exist_ok=True)
    existing = len(list(folder.glob("frame_*.jpg")))
    got, last_mtime = 0, None
    deadline = time.time() + max(90, n * interval * 3)
    print(f"Capturing {n} frames from {LIVE_FRAME} (Unity must be in Play mode)...")
    while got < n and time.time() < deadline:
        if LIVE_FRAME.exists():
            mtime = LIVE_FRAME.stat().st_mtime
            if mtime != last_mtime:
                try:
                    data = LIVE_FRAME.read_bytes()
                    Image.open(io.BytesIO(data)).load()          # raises if the file is half-written
                    dest = folder / f"frame_{existing + got + 1:04d}.jpg"
                    dest.write_bytes(data)                       # exact bytes, no re-compression
                    got += 1
                    last_mtime = mtime
                    print(f"  saved {dest.name}  ({got}/{n})")
                except Exception:
                    pass                                         # try again on the next poll
        time.sleep(interval)
    if got < n:
        print(f"Stopped after {got} frame(s). Is Unity running and exporting to {LIVE_FRAME.parent}?")
    return got


# ----------------------------------------------------------------- reporting
def contact_sheet(df, folder, max_images=40, cols=5, thumb=(192, 144)):
    sub = df.head(max_images)
    rows = (len(sub) + cols - 1) // cols
    label_h = 30
    sheet = Image.new("RGB", (cols * thumb[0], rows * (thumb[1] + label_h)), (20, 24, 34))
    draw = ImageDraw.Draw(sheet)
    for i, rec in enumerate(sub.itertuples()):
        x, y = (i % cols) * thumb[0], (i // cols) * (thumb[1] + label_h)
        try:
            im = Image.open(folder / rec.file).convert("RGB").resize(thumb)
            sheet.paste(im, (x, y))
        except Exception:
            pass
        draw.text((x + 4, y + thumb[1] + 2), f"{rec.prediction} {rec.confidence_pct:.0f}%", fill=(230, 236, 245))
        draw.text((x + 4, y + thumb[1] + 15), f"{rec.file}  green {rec.green_pct:.0f}%", fill=(140, 153, 179))
    out = folder / "contact_sheet.png"
    sheet.save(out)
    return out


def main():
    ap = argparse.ArgumentParser(description="Classify Unity drone frames with the app's land-cover model.")
    ap.add_argument("--capture", type=int, default=0, help="grab N frames from live_data/live_frame.jpg first")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between polls while capturing")
    ap.add_argument("--folder", type=Path, default=DEFAULT_FOLDER, help="frames folder (default data/unity_frames)")
    ap.add_argument("--truth", type=Path, default=None, help="CSV with columns file,true_label (optional)")
    args = ap.parse_args()

    folder = args.folder
    if args.capture > 0:
        capture_frames(args.capture, args.interval, folder)

    files = sorted(
        p for p in folder.glob("*")
        if p.suffix.lower() in IMAGE_EXT and p.name != "contact_sheet.png"      # never re-read our own output
    ) if folder.exists() else []
    if not files:
        sys.exit(f"No images found in {folder}. Use --capture 40 while Unity is running, or pass --folder.")

    model, label_map = load_model()
    if model is None:
        print("WARNING: model files not found in models/. Using the rule-based fallback (same as the app).\n")
    else:
        print("Model classes:", ", ".join(label_map.get(c, str(c)) for c in model.classes_), "\n")

    rows = []
    for p in files:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"skip {p.name}: {e}")
            continue
        label, conf, second, second_conf = classify(img, model, label_map)
        rows.append({
            "file": p.name, "prediction": label, "confidence_pct": round(conf * 100, 1),
            "runner_up": second, "runner_up_pct": round(second_conf * 100, 1),
            "green_pct": round(green_ratio_pct(img), 1),
        })
    df = pd.DataFrame(rows)

    print(df.to_string(index=False))
    n = len(df)
    print("\n" + "=" * 60 + f"\nSUMMARY ({n} frames)\n" + "=" * 60)
    counts = df["prediction"].value_counts()
    for label, c in counts.items():
        print(f"  {label:<20} {c:>4}  ({100 * c / n:.0f}%)")
    print(f"\n  mean confidence : {df['confidence_pct'].mean():.0f}%")
    print(f"  low confidence (<60%): {(df['confidence_pct'] < 60).sum()} frame(s)")
    print(f"  green ratio     : min {df['green_pct'].min():.0f}%  mean {df['green_pct'].mean():.0f}%  max {df['green_pct'].max():.0f}%")
    flips = int((df["prediction"] != df["prediction"].shift()).sum() - 1) if n > 1 else 0
    print(f"  label changes between consecutive frames: {flips} of {max(n - 1, 1)}")

    notes = []
    if counts.iloc[0] / n > 0.9 and n >= 10:
        notes.append(f"{counts.index[0]} wins >90% of frames: the model barely distinguishes these scenes.")
    if df["confidence_pct"].mean() < 60:
        notes.append("Average confidence is low: these frames look unlike the model's training images.")
    if n > 1 and flips / (n - 1) > 0.3:
        notes.append("The label flips often between neighbouring frames: predictions are unstable.")
    if df["green_pct"].between(30, 70).mean() > 0.5:
        notes.append("Green ratio sits mid-range on many frames: the ground colour is close to the 'green' rule threshold.")

    if args.truth is not None:
        truth = pd.read_csv(args.truth)
        merged = df.merge(truth, on="file", how="inner")
        if merged.empty:
            print("\nTruth file has no matching 'file' names.")
        else:
            merged["correct"] = merged["prediction"] == merged["true_label"]
            acc = merged["correct"].mean() * 100
            print(f"\n  ACCURACY vs your labels: {acc:.0f}%  ({int(merged['correct'].sum())}/{len(merged)})")
            print(merged.groupby("true_label")["correct"].agg(["mean", "count"])
                  .rename(columns={"mean": "accuracy", "count": "frames"}).round(2).to_string())
            wrong = merged[~merged["correct"]][["file", "true_label", "prediction", "confidence_pct"]]
            if not wrong.empty:
                print("\n  Mistakes:\n" + wrong.to_string(index=False))
            if acc < 70:
                notes.append("Accuracy on your labelled frames is below 70%: retraining with simulated frames is advised.")
    else:
        notes.append("Without your own labels this is NOT an accuracy figure. Label ~20 frames per class in a "
                     "truth.csv (file,true_label) and re-run with --truth to measure it.")

    df.to_csv(folder / "classification_report.csv", index=False)
    sheet = contact_sheet(df, folder)
    print("\nNOTES")
    for line in notes or ["No obvious problems in these frames."]:
        print("  -", line)
    print(f"\nSaved: {folder / 'classification_report.csv'}\n       {sheet}")


if __name__ == "__main__":
    main()
