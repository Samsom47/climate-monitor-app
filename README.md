# SpacePoint – Drone-Based Climate Monitoring AI Tool

Educational website for climate & remote sensing learning.  
Built for the **SpacePoint Internship**.

It now also receives **live data from a Unity drone simulation**: sensor readings and a camera image from a simulated drone flying a waypoint mission over a small 3D environment.

---

## Quick Start (2 steps only)

```bash
# 1. Install packages
pip install -r requirements.txt

# 2. Start the website
streamlit run app.py
```

Then open the link shown in the terminal (usually `http://localhost:8501`).

> **Requires Streamlit 1.37 or newer** (the live dashboard uses auto-refreshing fragments).  
> Check with `streamlit --version`. Upgrade with `pip install -U streamlit`.

> **Important:** Running `generate_sample_data.py` is **optional**.  
> The app automatically creates sample images and sensor data if they are missing.

---

## What this project includes

| Task | Feature | Where |
|------|---------|-------|
| 1 | Climate Monitoring Use Cases | Home page |
| 2 | Image Classification (trained model) | Image Classification page |
| 3 | Vegetation Analysis (green ratio) | Vegetation Analysis page |
| 4 | Heat Island heatmap (sample, upload, or live camera frame) | Heat Island Tool page |
| 5 | Sensor Dashboard + CSV upload + **live simulation feed** | Sensor Dashboard page |
| 6 | Student Worksheet + Instructor Guide | Student Activity Pack page |
| Sec. 1 | NDVI-Style Educational Tool | NDVI-Style Tool page |
| Sec. 2 | Drone Data Report Generator (can use live data) | Report Generator page |
|  7 | **Unity drone simulation** feeding the website live | `SpacePointDroneSimulation/` |

Also included:
- Real trained RandomForest model (`models/`)
- Sample aerial images + sensor data
- Separate scripts in `notebooks/`
- Works on **laptop and mobile** browsers

---

## How to use

1. Open the website
2. Use the left sidebar to switch between tools
3. **Image Classification / Vegetation**
   - Upload your own image **or** try a sample
   - Sample dropdown automatically disables when you upload a file
4. **Sensor Dashboard**
   - Uses built-in simulation data by default
   - Or upload your own CSV (temperature, humidity, AQI, GPS, etc.)
   - Or choose **Live from simulation** to see the Unity drone feed (see below)
5. **Student Activity Pack**
   - Download the worksheet and use it in class

---

## Live drone simulation (Unity)

A quadcopter flies a waypoint mission in a small 3D scene. It passes through a normal area, a high-pollution zone and a shaded zone, and its sensor readings change accordingly. Unity simply **writes two files**; the website reads them. No MQTT, ROS or network setup is needed.

```
Unity simulation ──writes──▶ live_data/live_sensors.csv   (one row appended every 1–2 s)
                             live_data/live_frame.jpg     (same file overwritten every 1–2 s)
                                        │
                                        ▼
                             Streamlit app (Live mode)
```

### Run it

1. Open the Unity project `SpacePointDroneSimulation/` with **Unity 6 (6000.0.84f1)**.  
   A newer Unity 6 version also works: let Unity upgrade the project when it asks.
2. Open the scene `Assets/Scenes/DroneSimulation`.
3. In the Hierarchy select **SimulationManager**. On **Live Data Exporter**, set **Output Folder** to the full path of this project's `live_data` folder, for example:
   ```
   C:\Users\YourName\Documents\SpacePoint_Climate_Tool\live_data
   ```
4. Press **Play** in Unity.
5. Start the website (`streamlit run app.py`) and open  
   **📊 Sensor Dashboard → Live from simulation**.

You should see the camera image, temperature, humidity, pressure, AQI, light, speed, trend charts and a map around Al Ain update about once per second.

### Tips

- Set **Export Interval** to `1` on Live Data Exporter for the smoothest dashboard.
- Tick **Clear Csv On Play** for a clean `live_sensors.csv` at the start of every run.
- Do **not** open `live_sensors.csv` in Excel while the simulation runs (Excel locks the file and Unity cannot append).
- The Heat Island Tool (**Live camera frame**) and the Report Generator (**Use live_data**) can use the same live files.
- The mission loops automatically; untick **Loop Mission** on the Waypoints object to fly once.

### File format

`live_data/live_sensors.csv` must have exactly this header:

```
timestamp,temperature_c,humidity_pct,pressure_hpa,air_quality_aqi,light_lux,latitude,longitude,speed_mps
```

`live_data/live_frame.jpg` is a 640 × 480 JPG of the drone camera view.

> Any other program (a different simulator or a test script) can feed the website the same way, as long as it writes these two files.

### Important limitation

Live mode works **only when Unity and Streamlit run on the same computer** (or share the same `live_data` folder). On Streamlit Community Cloud there is no `live_data` folder receiving Unity's files, so deploy the cloud version without live mode, or run the live demo locally.

### No Unity available?

(Only if `tools/fake_drone_sim.py` is included in your copy.)

```bash
python tools/fake_drone_sim.py
```

This writes the same two files with fake data so you can test the dashboard.

---

## Project Structure

```
SpacePoint_Climate_Tool/
├── app.py                      ← Main website (run this)
├── requirements.txt
├── README.md
├── generate_sample_data.py     ← Optional
├── train_models.py             ← Optional (retrain model)
├── models/
│   ├── landcover_classifier.joblib
│   ├── label_map.joblib
│   └── feature_names.joblib
├── data/
│   ├── sample_images/
│   ├── sensor_data.csv
│   └── thermal_grid.npy
├── live_data/                  ← Written by Unity, read by the website
│   ├── live_sensors.csv
│   └── live_frame.jpg
├── notebooks/
│   ├── 01_vegetation_analysis.py
│   └── 02_image_classification.py
└── student_pack/
    ├── student_worksheet.txt
    ├── instructor_guide.txt
    └── ndvi_worksheet.txt

SpacePointDroneSimulation/      ← Unity project (separate folder)
└── Assets/
    ├── Scenes/DroneSimulation.unity
    └── Scripts/                ← DroneController, WaypointManager, SensorSimulator,
                                  LiveDataExporter, EnvironmentZone, ...
```

---

## Sharing the website

**Option 1 – Local network**  
Run `streamlit run app.py`. Other devices on the same Wi-Fi can open the link. Live mode works when Unity runs on the same computer as Streamlit.

**Option 2 – Streamlit Community Cloud (free)**  
1. Push this folder to GitHub  
2. Go to https://share.streamlit.io  
3. Deploy `app.py`  

Live simulation mode will not receive Unity data in the cloud (see the limitation above); all other tools work.

---

## Notes

- Student-level educational prototype
- Real trained RandomForest model for classification
- Vegetation analysis uses green-pixel calculation (easy to explain)
- Simulated sensor values and GPS are generated by the Unity project; they are not real measurements
- Fully covers all 6 primary tasks of the internship document
- Works on laptop and mobile

**Ready to use.**
