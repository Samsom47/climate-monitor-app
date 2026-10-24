"""
SpacePoint - Drone-Based Climate Monitoring AI Tool
Simple educational web application covering all internship tasks.
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import os
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import io
import base64
import joblib
import requests
from datetime import datetime
from io import StringIO
 
LIVE_DIR = "live_data"
LIVE_CSV = os.path.join(LIVE_DIR, "live_sensors.csv")
LIVE_FRAME = os.path.join(LIVE_DIR, "live_frame.jpg")

# Page config
st.set_page_config(
    page_title="SpacePoint Climate Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — AEROSCAN dark HUD style (visual only, no content changes)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0A0E17;
        --panel: #111826;
        --panel-2: #161F30;
        --line: #28324A;
        --text: #E7ECF5;
        --muted: #8B99B3;
        --dim: #5A6580;
        --veg: #5FD38D;
        --water: #4FC3F7;
        --hot: #FF6B4A;
        --warn: #FFC24B;
        --alert: #FF5D5D;
    }

    .stApp {
        background:
            radial-gradient(1200px 600px at 80% -10%, rgba(95,211,141,0.06), transparent 60%),
            radial-gradient(900px 500px at 0% 100%, rgba(79,195,247,0.06), transparent 60%),
            var(--bg) !important;
        color: var(--text) !important;
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    section[data-testid="stSidebar"] {
        background: var(--panel) !important;
        border-right: 1px solid var(--line) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--muted) !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 13px !important;
        padding: 6px 0 !important;
    }

    .main-header {
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--text) !important;
        letter-spacing: -0.3px;
        margin-bottom: 0.15rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: var(--water) !important;
        margin-top: 1rem;
    }

    div[data-testid="stMetric"] {
        background: var(--panel-2) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        padding: 12px 14px !important;
    }
    div[data-testid="stMetric"] label {
        color: var(--muted) !important;
        font-size: 11px !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-family: "IBM Plex Mono", ui-monospace, monospace !important;
        font-size: 1.35rem !important;
    }

    .stButton > button {
        background: var(--veg) !important;
        color: #062012 !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        width: 100%;
    }
    .stButton > button:hover { filter: brightness(1.08); }
    .stButton > button:disabled {
        opacity: 0.4 !important;
        cursor: not-allowed !important;
    }

    .stDownloadButton > button {
        background: transparent !important;
        color: var(--text) !important;
        border: 1px solid var(--line) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        width: 100%;
    }
    .stDownloadButton > button:hover {
        border-color: var(--veg) !important;
        color: var(--veg) !important;
    }

    .stTextInput input, .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] > div,
    .stDateInput input {
        background: var(--panel-2) !important;
        color: var(--text) !important;
        border: 1px solid var(--line) !important;
        border-radius: 6px !important;
    }
    .stFileUploader [data-testid="stBaseButton-secondary"],
section[data-testid="stFileUploader"] button,
div[data-testid="stFileUploader"] button {
    background: #161F30 !important;
    color: #E7ECF5 !important;
    border: 1px solid #28324A !important;
    border-radius: 6px !important;
}
.stFileUploader button:hover,
section[data-testid="stFileUploader"] button:hover,
div[data-testid="stFileUploader"] button:hover {
    border-color: #5FD38D !important;
    color: #5FD38D !important;
    background: #111826 !important;
}

    .stTabs [data-baseweb="tab"] {
        color: var(--muted) !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--veg) !important;
        border-bottom: 2px solid var(--veg) !important;
    }

    p, li, label { color: var(--text); }
    h1, h2, h3, h4 { color: var(--text) !important; }

    .stPlotlyChart {
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 6px;
        background: var(--panel);
    }

    .stAlert {
        border-radius: 6px !important;
        border: 1px solid var(--line) !important;
    }

    header[data-testid="stHeader"] { background: transparent !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    @media (max-width: 768px) {
        .main-header { font-size: 1.35rem; }
        div[data-testid="stMetric"] { min-width: 45%; }
    }
    img { max-width: 100% !important; height: auto !important; }

    .stAppDeployButton {
    display: none !important;
}
div[data-testid="stAppDeployButton"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# ====================== HELPER FUNCTIONS ======================

def calculate_green_ratio(image):
    """Simple vegetation detection using green pixel ratio (Task 3)."""
    img = np.array(image.convert("RGB"))
    # Green dominance: G > R and G > B, and G is reasonably high
    green_mask = (img[:, :, 1] > img[:, :, 0]) & (img[:, :, 1] > img[:, :, 2]) & (img[:, :, 1] > 60)
    green_pixels = np.sum(green_mask)
    total_pixels = img.shape[0] * img.shape[1]
    ratio = (green_pixels / total_pixels) * 100

    if ratio >= 40:
        level = "High Vegetation 🌿"
        color = "green"
    elif ratio >= 15:
        level = "Medium Vegetation 🌱"
        color = "orange"
    else:
        level = "Low Vegetation 🏜️"
        color = "red"

    # Create mask image
    mask_img = img.copy()
    mask_img[~green_mask] = mask_img[~green_mask] // 3   # darken non-green
    return ratio, level, color, Image.fromarray(mask_img), green_mask


@st.cache_resource
def load_classifier():
    """Load the trained land-cover model."""
    model_path = "models/landcover_classifier.joblib"
    label_path = "models/label_map.joblib"
    if os.path.exists(model_path) and os.path.exists(label_path):
        model = joblib.load(model_path)
        label_map = joblib.load(label_path)
        return model, label_map
    return None, None


def extract_features_from_image(image):
    """Extract the same 9 features used during training."""
    img = image.convert("RGB").resize((64, 64))
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


def classify_image(image):
    """Classify using the real trained RandomForest model (with rule-based fallback)."""
    model, label_map = load_classifier()

    if model is not None:
        feats = extract_features_from_image(image)
        pred = model.predict(feats)[0]
        # Get probability of the predicted class
        probs = model.predict_proba(feats)[0]
        conf = float(probs[pred])
        label = label_map.get(pred, "Unknown")
        return label, conf

    # Fallback rule-based (if model files missing)
    img = np.array(image.convert("RGB").resize((100, 100)))
    avg_r, avg_g, avg_b = img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()
    if avg_g > avg_r + 15 and avg_g > avg_b + 10:
        return "Vegetation", 0.75
    elif avg_b > avg_r + 20 and avg_b > avg_g + 10:
        return "Water", 0.80
    elif abs(avg_r - avg_g) < 20 and abs(avg_g - avg_b) < 20 and avg_r < 120:
        return "Buildings / Roads", 0.65
    elif avg_r > 100 and avg_g > 70 and avg_b < 90:
        return "Sand / Soil", 0.70
    else:
        return "Mixed / Other", 0.55


def load_sensor_data():
    """Load sensor data. If file missing, generate simple simulated data on the fly."""
    path = "data/sensor_data.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # Generate simple simulated data so generate_sample_data.py is optional
    np.random.seed(42)
    n = 80
    dates = pd.date_range("2025-06-01", periods=n, freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "temperature_c": np.round(28 + 6 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 1.5, n), 1),
        "humidity_pct": np.round(55 + 15 * np.sin(np.linspace(0, 3*np.pi, n)) + np.random.normal(0, 4, n), 1),
        "pressure_hpa": np.round(1010 + np.random.normal(0, 3, n), 1),
        "air_quality_aqi": np.clip(np.round(45 + 30 * np.sin(np.linspace(0, 2*np.pi, n)) + np.random.normal(0, 10, n), 0), 10, 200).astype(int),
        "light_lux": np.round(np.clip(8000 + 6000 * np.sin(np.linspace(0, 2*np.pi, n)) + np.random.normal(0, 500, n), 0, 12000), 0).astype(int),
        "latitude": 24.1985 + np.random.normal(0, 0.008, n),  # UAE University (Al Ain)
        "longitude": 55.6810 + np.random.normal(0, 0.008, n),
        "speed_mps": np.round(np.clip(4 + 3 * np.sin(np.linspace(0, 2*np.pi, n)) + np.random.normal(0, 0.5, n), 0, 12), 1),
    })
    return df


def load_thermal():
    """Load thermal grid. If missing, generate a simple simulated heat island."""
    path = "data/thermal_grid.npy"
    if os.path.exists(path):
        return np.load(path)

    # Generate simple simulated thermal data on the fly
    np.random.seed(7)
    base = np.linspace(32, 42, 50)
    temp_grid = np.outer(base, np.ones(50)) + np.random.normal(0, 1.5, (50, 50))
    temp_grid[18:32, 18:32] += 6   # hot urban center
    temp_grid[5:15, 30:45] -= 4    # cooler park
    return temp_grid

def load_live_sensors():
    """Live sensor rows (Report Generator). Cloud: latest rows from Supabase. Local: live_data/live_sensors.csv."""
    if live_backend() == "supabase":
        return live_sensor_df(500)
    path = os.path.join("live_data", "live_sensors.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        df = pd.read_csv(path)
        for col in ["timestamp", "time", "datetime", "date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df = df.rename(columns={col: "timestamp"})
                break
        return df
    except Exception:
        return None


def load_live_frame():
    """Latest drone frame as (PIL.Image or None, 'HH:MM:SS'). Works for both local files and Supabase."""
    data, stamp = read_live_frame_bytes()
    if data is None:
        return None, None
    try:
        return Image.open(io.BytesIO(data)).convert("RGB"), stamp
    except Exception:
        return None, None


def ensure_sample_images():
    """Create simple sample images if they don't exist (so generate_sample_data.py is optional)."""
    os.makedirs("data/sample_images", exist_ok=True)

    def make_image(filename, kind):
        path = f"data/sample_images/{filename}"
        if os.path.exists(path):
            return
        img = Image.new("RGB", (400, 300), color=(100, 100, 100))
        pixels = img.load()
        if kind == "vegetation":
            for x in range(400):
                for y in range(300):
                    g = 80 + (x * y) % 100
                    pixels[x, y] = (30, min(g, 180), 30)
        elif kind == "water":
            for x in range(400):
                for y in range(300):
                    b = 120 + (x + y) % 80
                    pixels[x, y] = (20, 60, min(b, 220))
        elif kind == "urban":
            for x in range(400):
                for y in range(300):
                    v = 70 + (x // 40 + y // 40) % 50
                    pixels[x, y] = (v, v, v + 10)
            # roads
            for x in range(400):
                for y in range(140, 170):
                    pixels[x, y] = (40, 40, 40)
        elif kind == "soil":
            for x in range(400):
                for y in range(300):
                    r = 140 + (x * y) % 40
                    pixels[x, y] = (min(r, 180), 100, 50)
        elif kind == "mixed":
            for x in range(400):
                for y in range(300):
                    if x < 200 and y < 150:
                        pixels[x, y] = (40, 150, 40)
                    elif x >= 200 and y < 150:
                        pixels[x, y] = (30, 90, 180)
                    elif x < 200:
                        pixels[x, y] = (150, 110, 60)
                    else:
                        pixels[x, y] = (100, 100, 110)
        img.save(path)

    make_image("vegetation_1.jpg", "vegetation")
    make_image("vegetation_2.jpg", "vegetation")
    make_image("water_1.jpg", "water")
    make_image("urban_1.jpg", "urban")
    make_image("soil_1.jpg", "soil")
    make_image("mixed_1.jpg", "mixed")


#  PART 2 - paste with the other helper functions
#          (e.g. just above the "# ====== SIDEBAR ======" line)
# ---------------------------------------------------------------------
def read_live_tail(path, n_rows=300):
    """Read only the last n_rows of a growing CSV (stays fast after hours of data)."""
    try:
        size = os.path.getsize(path)
        if size == 0:
            return None
        with open(path, "rb") as f:
            header = f.readline().decode("utf-8", errors="ignore")
            chunk = 130 * n_rows                      # rows are ~75 bytes; 130 is a safe margin
            if size <= chunk + len(header):
                body = f.read().decode("utf-8", errors="ignore")
            else:
                f.seek(size - chunk)
                f.readline()                           # drop the partial first line
                body = f.read().decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(header + body))
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna()                               # drops a half-written last line
        return df.tail(n_rows).reset_index(drop=True)
    except Exception:
        return None
 
 
def read_live_frame_bytes():
    """Return (jpeg_bytes, 'HH:MM:SS') for the latest drone frame, or (None, None)."""
    data, stamp, _ = _live_frame_core()
    return data, stamp


def _find_cols(df):
    low = {c: c.lower() for c in df.columns}
    pick = lambda test: next((c for c, l in low.items() if test(l)), None)
    return dict(
        temp=pick(lambda l: "temp" in l),
        hum=pick(lambda l: "humid" in l),
        aqi=pick(lambda l: "aqi" in l or "air" in l),
        light=pick(lambda l: "light" in l or "lux" in l),
        speed=pick(lambda l: "speed" in l),
        lat=pick(lambda l: "lat" in l),
        lon=pick(lambda l: "lon" in l),
    )
 
 
def _fmt(df, col, spec, unit=""):
    if col is None:
        return "N/A"
    v = df.iloc[-1][col]
    try:
        return format(float(v), spec) + unit
    except Exception:
        return str(v)
 
 
def render_metrics(df, ncols=5):
    """Metric cards + alerts for the latest row."""
    c = _find_cols(df)
    items = [
        ("Temperature", _fmt(df, c["temp"], ".1f", " °C")),
        ("Humidity", _fmt(df, c["hum"], ".0f", " %")),
        ("Air Quality (AQI)", _fmt(df, c["aqi"], ".0f")),
        ("Light", _fmt(df, c["light"], ",.0f", " lux")),
        ("Speed", _fmt(df, c["speed"], ".1f", " m/s")),
    ]
    for i in range(0, len(items), ncols):
        cols = st.columns(ncols)
        for col, (label, value) in zip(cols, items[i:i + ncols]):
            col.metric(label, value)
 
    latest = df.iloc[-1]
    if c["aqi"] and pd.api.types.is_numeric_dtype(df[c["aqi"]]):
        aqi_val = latest[c["aqi"]]
        if aqi_val > 100:
            st.error("⚠️ Air Quality is unhealthy!")
        elif aqi_val > 50:
            st.warning("⚠️ Moderate air quality – sensitive groups should be careful.")
        else:
            st.success("✅ Air quality is good.")
    if c["temp"] and pd.api.types.is_numeric_dtype(df[c["temp"]]) and latest[c["temp"]] > 35:
        st.warning("🌡️ High temperature alert!")
 
 
def render_tabs(df, key_prefix, show_download=True):
    """Trends / Map / Raw data tabs. key_prefix keeps widget keys unique."""
    c = _find_cols(df)
    tab1, tab2, tab3 = st.tabs(["📈 Trends", "🗺️ Map", "📋 Raw Data"])
 
    with tab1:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            default = numeric_cols.index(c["temp"]) if c["temp"] in numeric_cols else 0
            metric = st.selectbox("Select metric", numeric_cols, index=default, key=f"{key_prefix}_metric")
            plot_df = df.tail(60)
            x = "timestamp" if "timestamp" in plot_df.columns else None
            fig = px.line(plot_df, x=x, y=metric, title=f"{metric} (last {len(plot_df)} points)")
            fig.update_traces(line=dict(width=2))
            fig.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20),
                              xaxis_title="Time", uirevision=key_prefix)
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_trend")
        else:
            st.info("No numeric columns found for charting.")
 
    with tab2:
        if c["lat"] and c["lon"]:
            pts = df.dropna(subset=[c["lat"], c["lon"]]).tail(60)
            if len(pts) > 0:
                last = pts.iloc[-1]
                if c["aqi"] and pd.api.types.is_numeric_dtype(df[c["aqi"]]):
                    colors = ["#FF5D5D" if v > 100 else ("#FFC24B" if v > 50 else "#5FD38D") for v in pts[c["aqi"]]]
                else:
                    colors = "#5FD38D"
                new_api = hasattr(go, "Scattermap")           # Plotly >= 5.24
                Trace = go.Scattermap if new_api else go.Scattermapbox
                fig = go.Figure()
                fig.add_trace(Trace(lat=pts[c["lat"]], lon=pts[c["lon"]], mode="lines+markers",
                                    marker=dict(size=7, color=colors),
                                    line=dict(width=2, color="#4FC3F7"), name="Path"))
                fig.add_trace(Trace(lat=[last[c["lat"]]], lon=[last[c["lon"]]], mode="markers",
                                    marker=dict(size=16, color="#FF6B4A"), name="Drone"))
                center = dict(lat=float(last[c["lat"]]), lon=float(last[c["lon"]]))
                layout = dict(style="open-street-map", center=center, zoom=18)   # zoom 18 = drone path is visible
                fig.update_layout(**({"map": layout} if new_api else {"mapbox": layout}))
                fig.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                                  uirevision=f"{key_prefix}_map")
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_map")
                st.caption(f"Drone: {center['lat']:.6f}, {center['lon']:.6f}  ·  line = last {len(pts)} positions, "
                           "colour = AQI (green / orange / red)")
            else:
                st.info("No valid GPS points found.")
        else:
            st.info("No latitude/longitude columns found for map.")
 
    with tab3:
        st.dataframe(df.tail(50), use_container_width=True)
 
    if show_download:
        st.download_button("Download current data as CSV", df.to_csv(index=False),
                           "sensor_data.csv", key=f"{key_prefix}_dl")
 
 
def render_heat_proxy(img, key):
    """Teaching heat proxy from an RGB frame (same maths as before)."""
    arr = np.array(img.convert("RGB").resize((200, 150)))
    gray = arr.mean(axis=2)
    heat_proxy = (255 - gray) / 255.0 * 20 + 28
    fig = px.imshow(heat_proxy, color_continuous_scale="hot_r", labels=dict(color="Est. °C"))
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True, key=key)
    c1, c2 = st.columns(2)
    c1.metric("Est. max surface temp", f"{heat_proxy.max():.1f} °C")
    c2.metric("Est. min surface temp", f"{heat_proxy.min():.1f} °C")
    st.info("This is a teaching proxy from the RGB frame (not a real thermal camera).")


# ====================== LIVE DATA BACKEND (local files OR Supabase cloud) ======================
# Unity can feed the app two ways:
#   - local : Unity writes live_data/live_sensors.csv + live_frame.jpg on the same PC
#   - cloud : Unity uploads to Supabase (table "live_sensors" + storage bucket "drone-frames");
#             the deployed app reads from there. Needs SUPABASE_URL and SUPABASE_KEY in the app secrets.
SB_LIMIT = 120                      # rows fetched per refresh (charts only use the last 60)
SB_TABLE = "live_sensors"
SB_HIDDEN_COLS = ["id", "frame_url", "created_at"]


def _secret(name):
    """Read a secret from Streamlit secrets, falling back to environment variables."""
    try:
        val = st.secrets.get(name, "")
    except Exception:
        val = ""
    if not val:
        val = os.environ.get(name, "")
    return str(val).strip()


def supabase_configured():
    return bool(_secret("SUPABASE_URL")) and bool(_secret("SUPABASE_KEY"))


def live_backend():
    """'supabase' or 'local', following the sidebar choice (Auto = Supabase when secrets exist)."""
    choice = st.session_state.get("live_source_choice", "Auto")
    if choice == "Local files":
        return "local"
    if choice == "Cloud (Supabase)":
        return "supabase"
    return "supabase" if supabase_configured() else "local"


@st.cache_data(ttl=1, show_spinner=False)
def _sb_rows(url, key, limit):
    """Newest rows first. Cached for 1 s so every widget and every viewer shares one request."""
    try:
        r = requests.get(
            url.rstrip("/") + "/rest/v1/" + SB_TABLE,
            headers={"apikey": key, "Authorization": f"Bearer {key}"},
            params={"select": "*", "order": "id.desc", "limit": str(limit)},
            timeout=8,
        )
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


def sb_rows(limit=SB_LIMIT):
    """(rows newest-first, error-or-None)."""
    if not supabase_configured():
        return [], "SUPABASE_URL / SUPABASE_KEY are not set in the app secrets."
    return _sb_rows(_secret("SUPABASE_URL"), _secret("SUPABASE_KEY"), int(limit))


@st.cache_data(ttl=30, max_entries=6, show_spinner=False)
def _fetch_frame(url):
    """Download one camera frame. The URL contains a timestamp, so each frame is its own cache entry."""
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        return r.content if len(r.content) > 1000 else None
    except Exception:
        return None


def live_sensor_df(n_rows=SB_LIMIT):
    """Latest live sensor rows as a DataFrame (oldest -> newest) or None. Same columns for both backends."""
    if live_backend() == "supabase":
        rows, _ = sb_rows(n_rows)
        if not rows:
            return None
        df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.drop(columns=[c for c in SB_HIDDEN_COLS if c in df.columns])
        return df.dropna().reset_index(drop=True)
    return read_live_tail(LIVE_CSV, n_rows)


def _live_frame_core():
    """(jpeg_bytes, 'HH:MM:SS', frame_id) for the newest drone frame, or (None, None, None)."""
    if live_backend() == "supabase":
        rows, _ = sb_rows(SB_LIMIT)
        tried = 0
        for row in rows:                                   # newest first
            url = row.get("frame_url")
            if not url:
                continue
            data = _fetch_frame(url)
            if data:
                ts = pd.to_datetime(row.get("timestamp"), errors="coerce")
                return data, (f"{ts:%H:%M:%S}" if pd.notna(ts) else "?"), url
            tried += 1
            if tried >= 3:                                  # newest upload may not be readable yet
                break
        return None, None, None
    try:
        with open(LIVE_FRAME, "rb") as f:
            data = f.read()
        if len(data) < 1000:
            return None, None, None
        mtime = os.path.getmtime(LIVE_FRAME)
        return data, datetime.fromtimestamp(mtime).strftime("%H:%M:%S"), mtime
    except Exception:
        return None, None, None


def get_live_frame():
    """Latest frame as (PIL image, 'HH:MM:SS', frame_id) or (None, None, None)."""
    data, stamp, frame_id = _live_frame_core()
    if data is None:
        return None, None, None
    try:
        return Image.open(io.BytesIO(data)).convert("RGB"), stamp, frame_id   # fails on a half-written file
    except Exception:
        return None, None, None


def live_feed_age():
    """Seconds since the newest sample arrived, or None if unknown."""
    try:
        if live_backend() == "supabase":
            rows, _ = sb_rows(SB_LIMIT)
            if not rows:
                return None
            ts = pd.to_datetime(rows[0].get("created_at"), utc=True, errors="coerce")
            if pd.isna(ts):
                return None
            return max(0.0, (pd.Timestamp.now(tz="UTC") - ts).total_seconds())
        if os.path.exists(LIVE_CSV):
            return max(0.0, datetime.now().timestamp() - os.path.getmtime(LIVE_CSV))
    except Exception:
        pass
    return None


def feed_badge():
    """Short status text: is Unity still sending data?"""
    age = live_feed_age()
    if age is None:
        return "⚪ Live feed"
    if age < 10:
        return f"🟢 LIVE · updated {age:.0f}s ago"
    if age < 60:
        return f"🟠 DELAYED · updated {age:.0f}s ago"
    if age < 3600:
        return f"🔴 OFFLINE · Unity stopped sending {age/60:.0f} min ago"
    return "🔴 OFFLINE · no recent data from Unity"


def live_source_caption():
    if live_backend() == "supabase":
        return ("Reads the live feed from the cloud (Supabase): sensor rows and the latest drone camera frame "
                "uploaded by the Unity simulation.")
    return "Reads `live_data/live_sensors.csv` and `live_data/live_frame.jpg` written by the Unity simulation."


def live_waiting_message(kind="sensors"):
    what = "sensor data" if kind == "sensors" else "camera frame"
    if live_backend() == "supabase":
        if not supabase_configured():
            return "Cloud live mode needs SUPABASE_URL and SUPABASE_KEY in the app secrets."
        _, err = sb_rows(SB_LIMIT)
        msg = f"No live {what} in the cloud yet. Press Play in the Unity simulation (Supabase uploader enabled)."
        if err:
            msg += f"  \nCould not reach Supabase: {err}"
        return msg
    return f"No live {what} found in live_data/. Press Play in Unity (check the Output Folder path)."


# ---------- Live camera helpers (Image Classification + Vegetation pages) ----------
def classify_with_probs(image):
    """Like classify_image(), but also returns every class probability.
    Returns (label, confidence, {label: probability}, source_text)."""
    model, label_map = load_classifier()
    if model is not None:
        feats = extract_features_from_image(image)
        probs = model.predict_proba(feats)[0]
        names = [label_map.get(c, str(c)) for c in list(model.classes_)]
        best = int(np.argmax(probs))
        return names[best], float(probs[best]), dict(zip(names, [float(p) for p in probs])), "trained RandomForest model"
    label, conf = classify_image(image)
    return label, conf, {label: conf}, "rule-based fallback (model files missing)"


def push_history(key, frame_id, entry, max_len=60):
    """Append one row to a rolling history in session_state, once per NEW frame."""
    hist = st.session_state.setdefault(f"{key}_hist", [])
    if st.session_state.get(f"{key}_last_id") != frame_id:
        hist.append(entry)
        del hist[:-max_len]
        st.session_state[f"{key}_last_id"] = frame_id
    return hist


def clear_history(key):
    st.session_state.pop(f"{key}_hist", None)
    st.session_state.pop(f"{key}_last_id", None)


# ====================== SIDEBAR ======================
st.sidebar.title("🌍 SpacePoint")
st.sidebar.markdown("**Climate Monitoring AI Tool**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home & Use Cases",
        "🖼️ Image Classification",
        "🌿 Vegetation Analysis",
        "🔥 Heat Island Tool",
        "📊 Sensor Dashboard",
        "📝 Student Activity Pack",
        "📗 NDVI-Style Tool",
        "📄 Report Generator",
    ]
)

st.sidebar.markdown("---")

with st.sidebar.expander("⚙️ Live data source"):
    st.selectbox(
        "Source for live pages",
        ["Auto", "Cloud (Supabase)", "Local files"],
        key="live_source_choice",
        help="Auto uses Supabase when SUPABASE_URL and SUPABASE_KEY are set in the app secrets, "
             "otherwise the local live_data folder.",
    )
st.sidebar.caption("Live source: " + ("☁️ Cloud (Supabase)" if live_backend() == "supabase" else "💻 Local files"))


# Ensure sample images exist (generate_sample_data.py is optional)
ensure_sample_images()


# ====================== PAGES ======================

# ---------- HOME & USE CASES (Task 1) ----------
if page == "🏠 Home & Use Cases":
    st.markdown('<p class="main-header">Drone-Based Climate Monitoring AI Tool</p>', unsafe_allow_html=True)
    st.markdown("Educational tool for climate and remote sensing learning using drone imagery and sensor data.")

    st.markdown("### 🎯 Climate Monitoring Use Cases")
    st.write("These are the main problems this tool helps students explore:")

    use_cases = [
        ("🌿 Vegetation Health", "Monitor green cover and detect stress in plants using drone images."),
        ("🔥 Urban Heat Island", "Find hotter zones in cities caused by concrete and lack of greenery."),
        ("💧 Water Body Detection", "Identify rivers, lakes, and ponds from aerial images."),
        ("🗺️ Land Cover Classification", "Classify areas into vegetation, soil, water, buildings, roads."),
        ("🗑️ Waste Detection", "Spot illegal dumping or waste piles (future extension)."),
        ("☀️ Solar Panel Inspection", "Check condition of solar farms from above (future extension)."),
        ("🌫️ Air Quality Mapping", "Combine sensor data with location to understand pollution."),
        ("🌾 Crop Stress Observation", "Help farmers see which fields need water or attention."),
    ]

    cols = st.columns(2)
    for i, (title, desc) in enumerate(use_cases):
        with cols[i % 2]:
            st.markdown(f"**{title}**  \n{desc}")

    st.markdown("---")
    st.success("✅ This website receives simulation / sensor data and shows all educational deliverables in one place.")


# ---------- IMAGE CLASSIFICATION (Task 2) ----------
elif page == "🖼️ Image Classification":
    st.markdown('<p class="main-header">Image Classification Demo</p>', unsafe_allow_html=True)
    st.write("Upload a drone/aerial image, try a sample, or use the live Unity camera frame.")

    clf_source = st.radio(
        "Image source",
        ["Upload image", "Sample image", "Live camera frame"],
        horizontal=True,
        key="clf_source"
    )

    samples = {
        "Vegetation": "data/sample_images/vegetation_1.jpg",
        "Water": "data/sample_images/water_1.jpg",
        "Urban": "data/sample_images/urban_1.jpg",
        "Soil": "data/sample_images/soil_1.jpg",
        "Mixed": "data/sample_images/mixed_1.jpg",
    }

    if clf_source == "Live camera frame":
        st.caption(live_source_caption() + " Only this section refreshes, so the page does not flicker.")
        k1, k2, k3 = st.columns([1, 2, 1])
        with k1:
            clf_auto = st.toggle("Auto-refresh", value=True, key="clf_live_auto")
        with k2:
            clf_every = st.select_slider("Refresh every (seconds)", options=[1, 2, 3, 5], value=2, key="clf_live_every")
        with k3:
            st.write("")
            if st.button("Clear history", key="clf_live_clear"):
                clear_history("clf_live")

        @st.fragment(run_every=(clf_every if clf_auto else None))
        def live_classify():
            img, stamp, frame_id = get_live_frame()
            if img is None:
                st.warning(live_waiting_message("frame"))
                return

            label, conf, probs, source = classify_with_probs(img)
            ratio, _, _, _, _ = calculate_green_ratio(img)
            hist = push_history("clf_live", frame_id, {
                "time": stamp, "prediction": label,
                "confidence_%": int(round(conf * 100)), "green_%": round(float(ratio), 1),
            })

            st.caption(f"{feed_badge()} · frame time {stamp}")
            left, right = st.columns([3, 2])
            with left:
                st.image(img, caption=f"Drone camera · updated {stamp}", use_container_width=True)
            with right:
                st.markdown(f"### Prediction: **{label}**")
                st.progress(min(max(conf, 0.0), 1.0))
                st.caption(f"Confidence: {conf*100:.0f}% ({source})")
                prob_df = pd.DataFrame({"class": list(probs.keys()), "probability": list(probs.values())})
                prob_df = prob_df.sort_values("probability")
                fig = px.bar(prob_df, x="probability", y="class", orientation="h", range_x=[0, 1])
                fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=None)
                st.plotly_chart(fig, use_container_width=True, key="clf_live_probs")

            if conf < 0.6:
                st.info("Low confidence: this frame does not look much like the images the model was trained on.")

            st.markdown("**Recent predictions**")
            counts = pd.Series([h["prediction"] for h in hist]).value_counts()
            st.caption(f"Last {len(hist)} frames: " + "  ·  ".join(f"{k}: {v}" for k, v in counts.items()))
            st.dataframe(pd.DataFrame(hist[-10:][::-1]), use_container_width=True, hide_index=True)

        live_classify()
        st.caption(
            "The simulated frames are stylised 3D renders, not real aerial photos, so the labels are indicative only."
        )
    else:
        image = None
        caption = ""

        col1, col2 = st.columns(2)
        with col1:
            if clf_source == "Upload image":
                uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="clf_upload")
                if uploaded is not None:
                    image = Image.open(uploaded)
                    caption = "Uploaded Image"
            else:
                sample_choice = st.selectbox("Select sample", list(samples.keys()), key="clf_sample")
                image = Image.open(samples[sample_choice])
                caption = f"Sample: {sample_choice}"

        with col2:
            if image:
                st.image(image, caption=caption, use_container_width=True)
                label, conf = classify_image(image)
                st.markdown(f"### Prediction: **{label}**")
                st.progress(conf)
                st.caption(f"Confidence: {conf*100:.0f}% (trained RandomForest model)")

    st.markdown("---")
    st.info("""
    **How it works:**  
    A real machine learning model (RandomForest) is used.  
    The image is converted into 9 simple features (color averages, texture, green/blue ratios).  
    The trained model then predicts one of these classes:
    - Vegetation  
    - Water  
    - Sand / Soil  
    - Buildings / Roads  
    - Mixed / Other  

    Model file: `models/landcover_classifier.joblib`
    """)


# ---------- VEGETATION ANALYSIS (Task 3) ----------
elif page == "🌿 Vegetation Analysis":
    st.markdown('<p class="main-header">Vegetation Analysis Tool</p>', unsafe_allow_html=True)
    st.write("Detect green areas, calculate green pixel ratio, and classify vegetation level.")

    veg_source = st.radio(
        "Image source",
        ["Upload image", "Sample image", "Live camera frame"],
        horizontal=True,
        key="veg_source"
    )

    samples = {
        "High Vegetation": "data/sample_images/vegetation_1.jpg",
        "Mixed Area": "data/sample_images/mixed_1.jpg",
        "Urban (Low)": "data/sample_images/urban_1.jpg",
        "Soil (Low)": "data/sample_images/soil_1.jpg",
    }

    if veg_source == "Live camera frame":
        st.caption(live_source_caption() + " Only this section refreshes, so the page does not flicker.")
        v1, v2, v3 = st.columns([1, 2, 1])
        with v1:
            veg_auto = st.toggle("Auto-refresh", value=True, key="veg_live_auto")
        with v2:
            veg_every = st.select_slider("Refresh every (seconds)", options=[1, 2, 3, 5], value=2, key="veg_live_every")
        with v3:
            st.write("")
            if st.button("Clear history", key="veg_live_clear"):
                clear_history("veg_live")

        @st.fragment(run_every=(veg_every if veg_auto else None))
        def live_vegetation():
            img, stamp, frame_id = get_live_frame()
            if img is None:
                st.warning(live_waiting_message("frame"))
                return

            ratio, level, color, mask_img, _ = calculate_green_ratio(img)
            hist = push_history("veg_live", frame_id, {"time": stamp, "green_pct": round(float(ratio), 1)})
            delta = None
            if len(hist) > 1:
                delta = f"{hist[-1]['green_pct'] - hist[-2]['green_pct']:+.1f}%"

            st.caption(f"{feed_badge()} · frame time {stamp}")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Green Pixel Ratio", f"{ratio:.1f}%", delta=delta, delta_color="off")
            with m2:
                st.markdown(f"### Classification: :{color}[{level}]")

            c1, c2 = st.columns(2)
            with c1:
                st.image(img, caption=f"Live frame · updated {stamp}", use_container_width=True)
            with c2:
                st.image(mask_img, caption="Green Areas Highlighted", use_container_width=True)

            if len(hist) > 1:
                st.markdown("**Green cover over recent frames**")
                fig = px.line(pd.DataFrame(hist), x="time", y="green_pct", markers=True)
                fig.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=20),
                                  yaxis=dict(title="Green %", range=[0, 100]), xaxis_title="Frame time")
                st.plotly_chart(fig, use_container_width=True, key="veg_live_trend")

        live_vegetation()
        st.caption(
            "The green rule counts any pixel where green is the strongest channel and above 60, so a green-grey "
            "ground texture also counts as vegetation. Treat the numbers as a demo of the method, not a survey."
        )
    else:
        image = None

        col1, col2 = st.columns(2)
        with col1:
            if veg_source == "Upload image":
                uploaded = st.file_uploader("Upload aerial image", type=["jpg", "jpeg", "png"], key="veg")
                if uploaded is not None:
                    image = Image.open(uploaded)
            else:
                sample_choice = st.selectbox("Select sample", list(samples.keys()), key="veg_sample")
                image = Image.open(samples[sample_choice])

        if image:
            ratio, level, color, mask_img, _ = calculate_green_ratio(image)

            with col2:
                st.metric("Green Pixel Ratio", f"{ratio:.1f}%")
                st.markdown(f"### Classification: :{color}[{level}]")

            c1, c2 = st.columns(2)
            with c1:
                st.image(image, caption="Original", use_container_width=True)
            with c2:
                st.image(mask_img, caption="Green Areas Highlighted", use_container_width=True)

            st.markdown("---")
            st.write("**Compare two images**")
            img2 = st.file_uploader("Upload second image for comparison", type=["jpg", "jpeg", "png"], key="veg2")
            if img2:
                image2 = Image.open(img2)
                ratio2, level2, _, _, _ = calculate_green_ratio(image2)
                diff = ratio - ratio2
                st.write(f"Image 1: **{ratio:.1f}%** ({level})  |  Image 2: **{ratio2:.1f}%** ({level2})")
                if abs(diff) < 3:
                    st.info("Both images have similar vegetation levels.")
                elif diff > 0:
                    st.success(f"Image 1 has **{diff:.1f}% more** green cover.")
                else:
                    st.warning(f"Image 2 has **{abs(diff):.1f}% more** green cover.")


# ---------- HEAT ISLAND TOOL (Task 4) ----------
elif page == "🔥 Heat Island Tool":
    st.markdown('<p class="main-header">Urban Heat Island Activity Tool</p>', unsafe_allow_html=True)
    st.write("Visualize surface temperatures and identify hotter zones (thermal data or image-based proxy).")

    st.subheader("Image source")
    heat_source = st.radio(
        "Choose source",
        ["Simulated thermal grid", "Upload image", "Live camera frame"],
        horizontal=True,
        key="heat_source"
    )

    image_for_heat = None

    if heat_source == "Upload image":
        up = st.file_uploader("Upload aerial image", type=["jpg", "jpeg", "png"], key="heat_upload")
        if up is not None:
            image_for_heat = Image.open(up)

    elif heat_source == "Live camera frame":
        heat_every = st.select_slider("Refresh every (seconds)", options=[1, 2, 3, 5], value=2, key="heat_every")
 
        @st.fragment(run_every=heat_every)
        def live_heat():
            data, t = read_live_frame_bytes()
            if data is None:
                st.warning(live_waiting_message("frame"))
                return
            img = Image.open(io.BytesIO(data)).convert("RGB")
            st.success(f"Live frame · updated {t}")
            st.image(img, caption="live_frame.jpg", use_container_width=True)
            render_heat_proxy(img, key="heat_live_chart")
 
        live_heat()

    if heat_source == "Simulated thermal grid":
        thermal = load_thermal()
        if thermal is not None:
            st.subheader("Surface Temperature Heatmap (°C)")
            fig = px.imshow(
                thermal,
                color_continuous_scale="hot_r",
                labels=dict(color="Temp °C"),
                aspect="auto"
            )
            fig.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

            max_temp = thermal.max()
            min_temp = thermal.min()
            mean_temp = thermal.mean()
            hot_threshold = mean_temp + 3

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Max Temp", f"{max_temp:.1f} °C")
            c2.metric("Min Temp", f"{min_temp:.1f} °C")
            c3.metric("Average", f"{mean_temp:.1f} °C")
            c4.metric("Hot Zones (> avg+3)", f"{np.sum(thermal > hot_threshold)} cells")

            st.markdown("---")
            st.subheader("What is Urban Heat Island (UHI)?")
            st.markdown("""
            Cities are often warmer than surrounding rural areas because:
            - Concrete and asphalt absorb and store heat
            - Fewer trees and green spaces
            - Waste heat from vehicles and air conditioners

            **In this map:** The bright center is a typical hot urban zone.  
            The cooler blue-ish area represents a park or water body.
            """)
            st.info("Students can discuss: Which areas need more trees? Where would you place cooling solutions?")
        else:
            st.error("Could not load thermal data.")

    elif image_for_heat is not None:
        st.subheader("Heat proxy from image (land-cover based estimate)")
        arr = np.array(image_for_heat.convert("RGB").resize((200, 150)))
        gray = arr.mean(axis=2)
        heat_proxy = (255 - gray) / 255.0 * 20 + 28
        fig = px.imshow(heat_proxy, color_continuous_scale="hot_r", labels=dict(color="Est. °C"))
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Est. max surface temp", f"{heat_proxy.max():.1f} °C")
        c2.metric("Est. min surface temp", f"{heat_proxy.min():.1f} °C")
        st.info("This is a teaching proxy from the RGB frame (not a real thermal camera).")


# ---------- SENSOR DASHBOARD (Task 5) ----------

elif page == "📊 Sensor Dashboard":
    st.markdown('<p class="main-header">Environmental Sensor Dashboard</p>', unsafe_allow_html=True)
    st.write("Dashboard that receives sensor / simulation data. Use built-in data, upload a CSV, or connect a live simulation feed.")
 
    os.makedirs(LIVE_DIR, exist_ok=True)
 
    st.subheader("Data Source")
    data_mode = st.radio(
        "Choose source",
        ["Built-in sample", "Upload CSV", "Live from simulation"],
        horizontal=True,
        key="sensor_data_mode"
    )
 
    if data_mode == "Live from simulation":
        st.caption(
            live_source_caption() + " "
            "Only the live section below refreshes, so the page does not flicker."
        )
        c_a, c_b = st.columns(2)
        with c_a:
            auto = st.toggle("Auto-refresh", value=True, key="live_auto")
        with c_b:
            every = st.select_slider("Refresh every (seconds)", options=[0.5, 1, 2, 3], value=1, key="live_every")
 
        @st.fragment(run_every=(every if auto else None))
        def live_view():
            df = live_sensor_df(SB_LIMIT)
            frame, frame_time = read_live_frame_bytes()
 
            if df is None or len(df) == 0:
                st.warning(live_waiting_message("sensors"))
                return
 
            last_ts = df["timestamp"].iloc[-1] if "timestamp" in df.columns else None
            stamp = f"{last_ts:%H:%M:%S}" if last_ts is not None and pd.notna(last_ts) else "?"
            st.caption(f"{feed_badge()} · {len(df)} recent rows · last sample {stamp} · page time {datetime.now():%H:%M:%S}")
 
            left, right = st.columns([2, 3])
            with left:
                if frame is not None:
                    st.image(frame, caption=f"Drone camera · updated {frame_time}", use_container_width=True)
                else:
                    st.info("No `live_frame.jpg` yet.")
            with right:
                render_metrics(df, ncols=3)
 
            render_tabs(df, key_prefix="live", show_download=False)
 
        live_view()
 
    else:
        df = None
        source = "Built-in sample"
        if data_mode == "Upload CSV":
            uploaded_csv = st.file_uploader(
                "Upload your own sensor CSV",
                type=["csv"],
                key="sensor_csv",
                help="CSV should have columns like: timestamp, temperature_c, humidity_pct, air_quality_aqi, latitude, longitude..."
            )
            if uploaded_csv is not None:
                try:
                    df = pd.read_csv(uploaded_csv)
                    for col in ["timestamp", "time", "datetime", "date"]:
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                            df = df.rename(columns={col: "timestamp"})
                            break
                    source = "Uploaded CSV"
                    st.success(f"Loaded uploaded CSV ({len(df)} rows)")
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")
                    df = load_sensor_data()
                    source = "Built-in sample (fallback)"
            else:
                st.info("Upload a CSV file to continue, or switch data source.")
        else:
            df = load_sensor_data()
 
        if df is not None and len(df) > 0:
            st.caption(f"Currently using: **{source}**")
            render_metrics(df, ncols=5)
            st.markdown("---")
            render_tabs(df, key_prefix="static", show_download=True)


# ---------- STUDENT ACTIVITY PACK (Task 6) ----------
elif page == "📝 Student Activity Pack":
    st.markdown('<p class="main-header">Student Activity Pack</p>', unsafe_allow_html=True)

    st.subheader("Student Worksheet")
    st.markdown("""
    **Activity Title:** Exploring Climate with Drones & Sensors

    **Instructions:** Use the tools in this website to answer the questions below.

    1. **What did the drone observe?**  
       (Look at the Image Classification or Vegetation pages. Describe what you see in one sample image.)

    2. **Which area is hotter?**  
       (Go to the Heat Island Tool. Identify the hottest zone and explain why it might be hotter.)

    3. **Which area has more vegetation?**  
       (Use the Vegetation Analysis tool. Compare two images and write the green ratios.)

    4. **What environmental problem can be identified?**  
       (Examples: low vegetation in urban areas, high temperature zones, poor air quality from the dashboard.)

    5. **What action can be recommended?**  
       (Suggest one practical action – plant trees, create parks, reduce traffic, etc.)

    ---
    **Bonus:** Record the latest temperature, humidity and AQI from the Sensor Dashboard.
    """)

    st.markdown("---")
    st.subheader("Instructor Guide")
    st.markdown("""
    **Learning Objectives**
    - Understand basic land-cover classification from aerial images
    - Calculate and interpret vegetation cover
    - Explain the urban heat island effect
    - Read simple environmental sensor data
    - Link observations to real climate actions

    **Suggested Time:** 45–60 minutes  
    **Materials:** Laptop / mobile + this website

    **Discussion Points**
    - Why do cities get hotter than countryside?
    - How can drones help climate education?
    - What simple actions can students take in their own neighborhood?

    **Answer Guidance**
    - Hotter zones = urban/concrete areas
    - More vegetation = higher green pixel ratio
    - Problems: heat, low greenery, moderate/poor AQI
    - Actions: tree planting, green roofs, reduce emissions
    """)

    # Simple downloadable text version
    worksheet_text = """SpacePoint Student Worksheet - Climate Monitoring with Drones

1. What did the drone observe?
________________________________________________

2. Which area is hotter?
________________________________________________

3. Which area has more vegetation? (write green ratios)
________________________________________________

4. What environmental problem can be identified?
________________________________________________

5. What action can be recommended?
________________________________________________

Bonus: Latest Temperature: _____ °C   Humidity: _____ %   AQI: _____
"""
    st.download_button("📥 Download Student Worksheet (TXT)", worksheet_text, "student_worksheet.txt")


# ---------- NDVI-STYLE TOOL (Secondary Project 1) ----------
elif page == "📗 NDVI-Style Tool":
    st.markdown('<p class="main-header">NDVI-Style Educational Tool</p>', unsafe_allow_html=True)
    st.write("Simplified vegetation index analysis for students (no special multispectral camera needed).")

    # --- Concept explanation ---
    st.subheader("1. What is NDVI?")
    st.markdown("""
    **NDVI** = Normalized Difference Vegetation Index  

    Real formula (used by satellites and drones with special cameras):
    """)
    st.latex(r"NDVI = \frac{NIR - Red}{NIR + Red}")
    st.markdown("""
    - **NIR** = Near-Infrared light (plants reflect a lot of it)
    - **Red** = Red light (plants absorb it for photosynthesis)

    Healthy plants → high NDVI (close to +1)  
    Bare soil / water / buildings → low NDVI (near 0 or negative)
    """)

    st.subheader("2. RGB vs Multispectral – Important Limitation")
    st.markdown("""
    | Type | What it captures | Can calculate real NDVI? |
    |------|------------------|--------------------------|
    | Normal RGB camera (phone/drone) | Red, Green, Blue only | ❌ No |
    | Multispectral camera | Red + Near-Infrared + more | ✅ Yes |

    **Our limitation:** Most student drones and phones only have RGB cameras.  
    So we create a **simplified Green Index** that works with normal photos.
    """)

    st.subheader("3. Simplified Green Index Tool")
    st.markdown("""
    Instead of real NDVI we use this student-friendly formula:
    """)
    st.latex(r"Green\ Index = \frac{G - R}{G + R + B + \epsilon}")
    st.caption("G = Green, R = Red, B = Blue channel. Higher value ≈ more green vegetation.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_ndvi = st.file_uploader("Upload aerial / drone image", type=["jpg", "jpeg", "png"], key="ndvi_upload")
        samples_ndvi = {
            "High Vegetation": "data/sample_images/vegetation_1.jpg",
            "Mixed Area": "data/sample_images/mixed_1.jpg",
            "Urban (Low)": "data/sample_images/urban_1.jpg",
            "Soil (Low)": "data/sample_images/soil_1.jpg",
            "Water": "data/sample_images/water_1.jpg",
        }
        sample_ndvi = st.selectbox(
            "Or choose sample",
            list(samples_ndvi.keys()),
            disabled=(uploaded_ndvi is not None),
            key="ndvi_sample"
        )
        if uploaded_ndvi is not None:
            st.caption("Sample dropdown disabled while an image is uploaded.")

    image_ndvi = None
    if uploaded_ndvi is not None:
        image_ndvi = Image.open(uploaded_ndvi)
    else:
        image_ndvi = Image.open(samples_ndvi[sample_ndvi])

    if image_ndvi:
        arr = np.array(image_ndvi.convert("RGB")).astype(np.float32)
        R = arr[:, :, 0]
        G = arr[:, :, 1]
        B = arr[:, :, 2]
        # Simplified green index (similar spirit to NDVI)
        green_index = (G - R) / (G + R + B + 1e-5)
        # Normalize for display
        gi_display = (green_index - green_index.min()) / (green_index.max() - green_index.min() + 1e-5)

        mean_gi = float(green_index.mean())
        if mean_gi > 0.15:
            veg_label = "High vegetation signal 🌿"
            veg_color = "green"
        elif mean_gi > 0.05:
            veg_label = "Medium vegetation signal 🌱"
            veg_color = "orange"
        else:
            veg_label = "Low vegetation signal 🏜️"
            veg_color = "red"

        with col2:
            st.metric("Mean Green Index", f"{mean_gi:.3f}")
            st.markdown(f"### Result: :{veg_color}[{veg_label}]")

        c1, c2 = st.columns(2)
        with c1:
            st.image(image_ndvi, caption="Original Image", use_container_width=True)
        with c2:
            # Visual output – colormap of green index
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(green_index, cmap="RdYlGn", vmin=-0.3, vmax=0.5)
            ax.set_title("Green Index Map (red = low, green = high)")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")
    st.subheader("4. Student-Friendly Summary")
    st.info("""
    **Remember:**
    - Real NDVI needs a special camera that sees Near-Infrared light.
    - With a normal RGB photo we can still estimate “how green” an area is.
    - This Green Index is perfect for classroom experiments with ordinary drones or phone photos.
    - Higher Green Index ≈ more healthy vegetation.
    """)

    # Worksheet download
    ndvi_worksheet = """NDVI-Style Educational Tool – Student Worksheet
================================================

1. What does NDVI stand for?
________________________________________________

2. Why can't a normal phone camera calculate real NDVI?
________________________________________________

3. What is the difference between RGB and multispectral cameras?
________________________________________________

4. Upload or select an image. What Mean Green Index did you get?
   Value: ________     Interpretation: ________________

5. Looking at the Green Index map, which parts of the image look healthiest?
________________________________________________

6. One real-world use of vegetation indices is:
________________________________________________
"""
    st.download_button("📥 Download NDVI Student Worksheet", ndvi_worksheet, "ndvi_worksheet.txt")


# ---------- REPORT GENERATOR (Secondary Project 2) ----------

# ---------- REPORT GENERATOR (Secondary Project 2) ----------
elif page == "📄 Report Generator":
    st.markdown('<p class="main-header">Drone Data Report Generator</p>', unsafe_allow_html=True)
    st.write("Upload images and/or sensor CSV → automatically generate a simple mission report.")

    for key, default in [
        ("report_text", None),
        ("report_sensor_df", None),
        ("report_mission_name", "Demo Climate Survey"),
        ("saved_report_images", None),
        ("saved_report_csv", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.subheader("0. Quick fill from live simulation")
    use_live = st.checkbox("Use live_data (sensors + camera frame)", key="report_use_live")
    if use_live:
        live_df = load_live_sensors()
        live_img, live_mtime = load_live_frame()
        if live_df is not None:
            st.success(f"Live sensors loaded ({len(live_df)} rows)")
            st.session_state.saved_report_csv = live_df.to_csv(index=False).encode("utf-8")
        else:
            st.warning(live_waiting_message("sensors"))
        if live_img is not None:
            st.success(f"Live camera frame loaded · {live_mtime}")
            st.image(live_img, caption="Live frame for report", use_container_width=True)
            buf = io.BytesIO()
            live_img.save(buf, format="JPEG")
            st.session_state.saved_report_images = [("live_frame.jpg", buf.getvalue())]
        else:
            st.warning(live_waiting_message("frame"))

    st.subheader("1. Upload Data")
    col_a, col_b = st.columns(2)
    with col_a:
        report_images = st.file_uploader(
            "Upload drone / aerial images (optional)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="report_imgs"
        )
    with col_b:
        report_csv = st.file_uploader(
            "Upload sensor CSV (optional)",
            type=["csv"],
            key="report_csv"
        )

    if report_images:
        st.session_state.saved_report_images = [(f.name, f.getvalue()) for f in report_images]
    if report_csv is not None:
        st.session_state.saved_report_csv = report_csv.getvalue()

    n_imgs = len(st.session_state.saved_report_images) if st.session_state.saved_report_images else 0
    has_csv = st.session_state.saved_report_csv is not None
    st.caption(f"Saved for report: **{n_imgs} image(s)** | CSV: **{'Yes' if has_csv else 'No'}**")

    st.subheader("2. Mission Info (optional)")
    mission_name = st.text_input("Mission / Location name", value="Demo Climate Survey")
    mission_date = st.date_input("Mission date")
    notes = st.text_area("Extra notes", value="Student educational flight.")

    report_exists = st.session_state.report_text is not None
    if report_exists:
        st.warning("A report already exists. Click **Clear Report** before generating a new one.")

    generate_clicked = st.button(
        "🚀 Generate Report",
        type="primary",
        key="btn_generate_report",
        disabled=report_exists
    )

    if generate_clicked and not report_exists:
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("SPACEPOINT – DRONE CLIMATE MISSION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Mission Name : {mission_name}")
        report_lines.append(f"Date         : {mission_date}")
        report_lines.append(f"Notes        : {notes}")
        report_lines.append("")

        report_lines.append("-" * 40)
        report_lines.append("1. IMAGE ANALYSIS")
        report_lines.append("-" * 40)

        saved_imgs = st.session_state.saved_report_images
        if saved_imgs:
            report_lines.append(f"Number of images: {len(saved_imgs)}")
            veg_ratios = []
            class_counts = {}
            for i, (img_name, img_bytes) in enumerate(saved_imgs, 1):
                img = Image.open(io.BytesIO(img_bytes))
                ratio, level, _, _, _ = calculate_green_ratio(img)
                label, conf = classify_image(img)
                veg_ratios.append(ratio)
                class_counts[label] = class_counts.get(label, 0) + 1
                report_lines.append(f"  Image {i}: {img_name}")
                report_lines.append(f"    → Land cover: {label} ({conf*100:.0f}%)")
                report_lines.append(f"    → Green ratio: {ratio:.1f}% ({level})")
            avg_veg = sum(veg_ratios) / len(veg_ratios)
            report_lines.append(f"\nAverage green cover: {avg_veg:.1f}%")
            report_lines.append("Land-cover summary:")
            for k, v in class_counts.items():
                report_lines.append(f"  - {k}: {v} image(s)")
        else:
            report_lines.append("No images provided.")

        report_lines.append("")
        report_lines.append("-" * 40)
        report_lines.append("2. SENSOR DATA SUMMARY")
        report_lines.append("-" * 40)

        sensor_df = None
        if st.session_state.saved_report_csv is not None:
            try:
                sensor_df = pd.read_csv(io.BytesIO(st.session_state.saved_report_csv))
                report_lines.append(f"Source: CSV / live feed ({len(sensor_df)} rows)")
            except Exception as e:
                report_lines.append(f"Could not read CSV: {e}")
        else:
            sensor_df = load_sensor_data()
            report_lines.append("Source: Built-in simulation data")

        if sensor_df is not None and len(sensor_df) > 0:
            numeric = sensor_df.select_dtypes(include=[np.number])
            report_lines.append("\nSummary statistics:")
            report_lines.append(numeric.describe().round(2).to_string())
            report_lines.append("")
            report_lines.append("-" * 40)
            report_lines.append("3. AI / AUTOMATED FINDINGS")
            report_lines.append("-" * 40)
            findings = []
            temp_col = next((c for c in sensor_df.columns if "temp" in c.lower()), None)
            aqi_col = next((c for c in sensor_df.columns if "aqi" in c.lower() or "air" in c.lower()), None)
            if temp_col and pd.api.types.is_numeric_dtype(sensor_df[temp_col]):
                t_mean = sensor_df[temp_col].mean()
                t_max = sensor_df[temp_col].max()
                findings.append(f"- Average temperature: {t_mean:.1f} °C (max {t_max:.1f} °C)")
                if t_max > 35:
                    findings.append("- High temperature values detected → possible heat stress risk.")
            if aqi_col and pd.api.types.is_numeric_dtype(sensor_df[aqi_col]):
                aqi_mean = sensor_df[aqi_col].mean()
                findings.append(f"- Average Air Quality Index: {aqi_mean:.0f}")
                if aqi_mean > 100:
                    findings.append("- Air quality is in the unhealthy range.")
                elif aqi_mean > 50:
                    findings.append("- Air quality is moderate.")
                else:
                    findings.append("- Air quality appears good.")
            if not findings:
                findings.append("- Basic statistics calculated.")
            for f in findings:
                report_lines.append(f)
        else:
            report_lines.append("No sensor data available.")

        report_lines.append("")
        report_lines.append("-" * 40)
        report_lines.append("4. RECOMMENDATIONS")
        report_lines.append("-" * 40)
        report_lines.append("- Continue monitoring green cover and temperature trends.")
        report_lines.append("- Areas with low vegetation and high temperature may benefit from tree planting.")
        report_lines.append("- Share results with students for discussion.")
        report_lines.append("")
        report_lines.append("=" * 60)
        report_lines.append("End of Report – Generated by SpacePoint Climate Tool")
        report_lines.append("=" * 60)

        st.session_state.report_text = "\n".join(report_lines)
        st.session_state.report_sensor_df = sensor_df
        st.session_state.report_mission_name = mission_name
        st.rerun()

    if st.session_state.report_text is not None:
        st.success("Report generated! Clear it before generating a new one.")
        st.text_area("Report Preview", st.session_state.report_text, height=450, key="report_preview")
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            st.download_button(
                "📥 Download Report (TXT)",
                st.session_state.report_text,
                file_name=f"SpacePoint_Report_{st.session_state.report_mission_name.replace(' ', '_')}.txt",
                mime="text/plain",
                key="report_download"
            )
        with btn_col2:
            if st.button("🗑️ Clear Report", key="clear_report"):
                st.session_state.report_text = None
                st.session_state.report_sensor_df = None
                st.session_state.report_mission_name = "Demo Climate Survey"
                st.session_state.saved_report_images = None
                st.session_state.saved_report_csv = None
                st.rerun()

        sensor_df = st.session_state.report_sensor_df
        if sensor_df is not None and len(sensor_df) > 0:
            st.subheader("Quick Charts")
            num_cols = sensor_df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                chart_col = st.selectbox("Chart column", num_cols, key="report_chart")
                if "timestamp" in sensor_df.columns:
                    fig = px.line(sensor_df, x="timestamp", y=chart_col, title=chart_col)
                else:
                    fig = px.line(sensor_df, y=chart_col, title=chart_col)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload data or use live_data, then click **Generate Report**.")

    # Auto-refresh when using live_data
    if use_live:
        auto_report = st.checkbox("Auto-refresh live data every 3 seconds", value=False, key="report_auto")
        if auto_report:
            import time as _time
            _time.sleep(3)
            st.rerun()

