"""
app.py
Main Streamlit dashboard for FlotAI.
Ties together database.py, cv_features.py, and ml_model.py into
the live/upload dashboard we designed in Step 5.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

from database import init_db, seed_default_data, save_reading, save_froth_features, get_recent_readings
from cv_features import extract_all_features
from ml_model import predict_recovery_grade

# --- One-time setup ---
init_db()  # safe to call every run - only creates tables if missing
seed_default_data()  # ensures plant_id=1 and created_by=1 exist for foreign keys

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

RECOVERY_ALERT_THRESHOLD = 70.0  # below this, we show a warning banner

st.set_page_config(page_title="FlotAI Control Room", layout="wide")


# --- Helper function: run the full pipeline on a video file ---
def process_video_and_predict(video_path):
    """
    Runs the complete pipeline: extract features -> predict -> save to DB.
    Returns a dict with everything the dashboard needs to display.
    """
    features = extract_all_features(video_path)
    prediction = predict_recovery_grade(features)

    reading_id = save_reading(
        image_path=str(video_path),
        predicted_recovery=prediction["predicted_recovery"],
        predicted_grade=prediction["predicted_grade"],
        confidence_score=None,  # placeholder - could add model confidence later
    )

    save_froth_features(
        reading_id=reading_id,
        avg_bubble_size=features["avg_bubble_size"],
        color_hue_avg=features["color_hue_avg"],
        texture_score=features["texture_score"],
        froth_speed=features["froth_speed"],
    )

    return {**features, **prediction, "reading_id": reading_id}


# --- Header ---
st.title("🏭 FlotAI Control Room")
st.caption("AI-powered froth flotation recovery & grade prediction")

# --- Toggle: Live vs Upload ---
mode = st.radio("Input source", ["Upload video/image sequence", "Use sample video (simulated live feed)"], horizontal=True)

video_path = None

if mode == "Upload video/image sequence":
    uploaded_file = st.file_uploader("Upload a short froth video clip (.mp4)", type=["mp4"])
    if uploaded_file is not None:
        # Save the uploaded file into our data/uploads folder with a
        # timestamped name, so multiple uploads never overwrite each other
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = UPLOAD_DIR / f"upload_{timestamp_str}.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {uploaded_file.name}")

else:
    # "Simulated live feed" - lets us demo without needing a real camera
    # connected during judging, by picking from our existing sample videos
    sample_videos = list(Path("videos").glob("*.mp4"))
    if sample_videos:
        selected = st.selectbox(
            "Select a sample clip to simulate a live camera feed",
            sample_videos,
            format_func=lambda p: p.name,
        )
        video_path = selected
    else:
        st.warning("No sample videos found in the 'videos' folder.")


# --- Run prediction when we have a video ---
if video_path is not None:
    if st.button("Analyze froth", type="primary"):
        with st.spinner("Extracting froth features and predicting recovery..."):
            try:
                result = process_video_and_predict(video_path)
                st.session_state["latest_result"] = result
            except Exception as e:
                st.error(f"Error processing video: {e}")


# --- Display latest result ---
if "latest_result" in st.session_state:
    result = st.session_state["latest_result"]

    st.divider()

    # Alert banner
    if result["predicted_recovery"] < RECOVERY_ALERT_THRESHOLD:
        st.warning(f"⚠️ Predicted recovery ({result['predicted_recovery']}%) is below the {RECOVERY_ALERT_THRESHOLD}% threshold.")

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted Recovery", f"{result['predicted_recovery']}%")
    col2.metric("Predicted Grade", f"{result['predicted_grade']}%")
    col3.metric("Avg Bubble Size", f"{result['avg_bubble_size']:.1f}")
    col4.metric("Froth Speed", f"{result['froth_speed']:.2f}")

    with st.expander("Raw extracted froth features"):
        st.json({
            "color_hue_avg": result["color_hue_avg"],
            "avg_bubble_size": result["avg_bubble_size"],
            "texture_score": result["texture_score"],
            "froth_speed": result["froth_speed"],
        })


# --- History / trend chart ---
st.divider()
st.subheader("📈 Recent readings")

rows = get_recent_readings(limit=50)

if rows:
    df = pd.DataFrame(rows, columns=["id", "timestamp", "predicted_recovery", "predicted_grade", "confidence_score"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    st.line_chart(df.set_index("timestamp")[["predicted_recovery", "predicted_grade"]])
    st.dataframe(df, use_container_width=True)
else:
    st.info("No readings yet. Analyze a video above to get started.")