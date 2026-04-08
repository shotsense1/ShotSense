import streamlit as st
import json
import pandas as pd
import os
import altair as alt
import cv2
import av
import tempfile
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
from detector_cloud import process_video

st.set_page_config(page_title="ShotSense Dashboard", layout="wide")


# ---------------- SESSION STATE ----------------
if "cloud_events" not in st.session_state:
    st.session_state.cloud_events = []

# ---------------- LOAD DATA ----------------
def load_data():
    data = st.session_state.cloud_events
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if df.empty:
        return df

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["practice_day"] = df["created_at"].dt.date
    df["practice_month"] = df["created_at"].dt.to_period("M").astype(str)
    df["shot_zone"] = "Field Goal"
    return df

df = load_data()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("ShotSense")

    player = st.text_input("Player Name", "Player 1")
    team = st.text_input("Team", "Training")

    st.markdown("---")
    st.subheader("Video Processing")

    session_name = st.text_input("Session Name", "Practice Day 1")

    uploaded_file = st.file_uploader(
        "Upload Basketball Video",
        type=["mp4", "mov", "m4v", "avi"]
    )

    # ✅ PREVIEW VIDEO (confirms upload works)
    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")
        st.video(uploaded_file)

    # ✅ FIXED BUTTON
    if st.button("Run Shot Detection"):
        if uploaded_file is None:
            st.error("Please upload a video first.")
        else:
            temp_path = None

            try:
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                if not ext:
                    ext = ".mp4"

                # ✅ FIX: use getbuffer()
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_file.flush()
                    temp_path = tmp_file.name

                # ✅ DEBUG CHECKS
                if not os.path.exists(temp_path):
                    st.error("File was not saved.")
                    st.stop()

                if os.path.getsize(temp_path) == 0:
                    st.error("Saved file is empty.")
                    st.stop()

                st.write(f"Temp file created: {temp_path}")

                # ✅ RUN DETECTION
                with st.spinner("Processing video..."):
                    shot_events = process_video(temp_path, session_name)

                if shot_events is None:
                    st.error("Detection returned None.")
                elif len(shot_events) == 0:
                    st.warning("Video processed but no shots detected.")
                else:
                    st.session_state.cloud_events.extend(shot_events)
                    st.success(f"Detected {len(shot_events)} shots!")
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {type(e).__name__} - {e}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    st.markdown("---")
    page = st.radio("Navigation", ["Profile", "Analytics", "Live"])

# ---------------- EMPTY STATE ----------------
if df.empty and page != "Live":
    st.title("ShotSense Dashboard")
    st.markdown("Upload a video and run detection to generate shot data.")
    st.stop()

# ---------------- STATS ----------------
if not df.empty:
    total_shots = len(df)
    makes = len(df[df["result"] == "MAKE"])
    misses = len(df[df["result"] == "MISS"])
    fg = (makes / total_shots * 100) if total_shots > 0 else 0

# ================= PROFILE =================
if page == "Profile":
    st.title("Profile")

    st.write(f"Shots: {total_shots}")
    st.write(f"Makes: {makes}")
    st.write(f"Misses: {misses}")
    st.write(f"FG%: {fg:.1f}%")

# ================= ANALYTICS =================
elif page == "Analytics":
    st.title("Analytics")

    if not df.empty:
        zone_counts = df.groupby(["shot_zone", "result"]).size().reset_index(name="count")

        chart = alt.Chart(zone_counts).mark_bar().encode(
            x="shot_zone",
            y="count",
            color="result"
        )

        st.altair_chart(chart, use_container_width=True)

# ================= LIVE =================
elif page == "Live":
    st.title("Live")

    RTC_CONFIG = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class LiveVideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=LiveVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )
