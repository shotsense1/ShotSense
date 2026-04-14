import streamlit as st
import json
import pandas as pd
import os
import altair as alt
import cv2
import av
import tempfile
import re
import time
import threading
import numpy as np
import inspect
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
from detector_cloud import process_video

st.set_page_config(page_title="ShotSense Dashboard", layout="wide")

# ---------------- NBA COLORS ----------------
BG_MAIN = "#0a0f1f"
BG_CARD = "#121a2f"
BG_CARD_2 = "#18233f"
TXT_MAIN = "#ffffff"
TXT_SUB = "#cbd5e1"
RED = "#c8102e"
BLUE = "#1d428a"
WHITE = "#f8fafc"
GREEN = "#22c55e"
BORDER = "rgba(255,255,255,0.08)"

# ---------------- DEFAULT ROI ----------------
DEFAULT_HOOP_ROI = (540, 395, 103, 37)
DEFAULT_NET_ROI = (563, 450, 48, 41)

# ---------------- SESSION STATE ----------------
if "cloud_events" not in st.session_state:
    st.session_state.cloud_events = []

if "HOOP_ROI" not in st.session_state:
    st.session_state.HOOP_ROI = DEFAULT_HOOP_ROI

if "NET_ROI" not in st.session_state:
    st.session_state.NET_ROI = DEFAULT_NET_ROI

if "detect_mode" not in st.session_state:
    st.session_state.detect_mode = False

if "motion_thresh" not in st.session_state:
    st.session_state.motion_thresh = 35

if "net_thresh" not in st.session_state:
    st.session_state.net_thresh = 28

if "min_area" not in st.session_state:
    st.session_state.min_area = 220

if "motion_frames_needed" not in st.session_state:
    st.session_state.motion_frames_needed = 3

if "cooldown_seconds" not in st.session_state:
    st.session_state.cooldown_seconds = 2.0

if "result_hold_seconds" not in st.session_state:
    st.session_state.result_hold_seconds = 1.2

if "calib_frame_bytes" not in st.session_state:
    st.session_state.calib_frame_bytes = None

# ---------------- CSS ----------------
st.markdown(f"""
<style>
    .stApp {{
        background: linear-gradient(180deg, {BG_MAIN} 0%, #111827 100%);
        color: {TXT_MAIN};
    }}

    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }}

    .section-card {{
        background: linear-gradient(135deg, {BG_CARD} 0%, {BG_CARD_2} 100%);
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
    }}

    .metric-card {{
        background: linear-gradient(135deg, {BG_CARD} 0%, {BG_CARD_2} 100%);
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
    }}

    .metric-label {{
        color: {TXT_SUB};
        font-size: 0.9rem;
        margin-bottom: 8px;
    }}

    .metric-value {{
        font-size: 1.8rem;
        font-weight: bold;
        color: {TXT_MAIN};
    }}

    .summary-box {{
        background: linear-gradient(135deg, {BLUE} 0%, {RED} 100%);
        border-radius: 18px;
        padding: 20px;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.25);
    }}

    .feed-item {{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
        color: {TXT_MAIN};
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid {BORDER};
    }}

    div[data-testid="stDataFrame"] {{
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid {BORDER};
    }}

    h1, h2, h3 {{
        color: {TXT_MAIN} !important;
    }}
</style>
""", unsafe_allow_html=True)

# ---------------- HELPERS ----------------
def metric(label, value):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)

def format_display_time(value):
    try:
        dt = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt):
            return value
        return dt.strftime("%Y-%m-%d %I:%M:%S %p")
    except Exception:
        return value

def clean_display_table(source_df):
    display_df = source_df.copy()

    for col in ["hoop_roi", "net_roi", "shot_location", "player_name"]:
        if col in display_df.columns:
            display_df = display_df.drop(columns=[col])

    if "created_at" in display_df.columns:
        display_df["created_at"] = display_df["created_at"].apply(format_display_time)

    display_df = display_df.rename(columns={
        "time_sec": "Shot Time (s)",
        "result": "Result",
        "shot_zone": "Shot Zone",
        "video_file": "Session Name",
        "created_at": "Recorded At",
        "practice_day": "Practice Day",
        "practice_month": "Practice Month"
    })
    return display_df

def safe_player_filename(player_name):
    cleaned = re.sub(r'[^a-zA-Z0-9_-]+', '_', player_name.strip())
    if not cleaned:
        cleaned = "Player_1"
    return f"data_{cleaned}.json"

def extract_player_name_from_filename(filename):
    base = os.path.basename(filename)
    if base.startswith("data_") and base.endswith(".json"):
        return base[5:-5].replace("_", " ")
    return base

def save_data(player_name, new_events):
    filename = safe_player_filename(player_name)

    for event in new_events:
        event["player_name"] = player_name

    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = []
    else:
        existing_data = []

    existing_data.extend(new_events)

    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=2)

def load_data(player_name):
    filename = safe_player_filename(player_name)

    if not os.path.exists(filename):
        return pd.DataFrame()

    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except Exception:
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if df.empty:
        return df

    if "player_name" not in df.columns:
        df["player_name"] = player_name

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    else:
        df["created_at"] = pd.to_datetime(datetime.now())

    df["practice_day"] = df["created_at"].dt.date
    df["practice_month"] = df["created_at"].dt.to_period("M").astype(str)

    if "shot_zone" not in df.columns:
        df["shot_zone"] = "Field Goal"
    else:
        df["shot_zone"] = df["shot_zone"].fillna("Field Goal")

    if "video_file" not in df.columns:
        df["video_file"] = "Unknown Session"

    return df

def load_all_users_data():
    files = [f for f in os.listdir(".") if f.startswith("data_") and f.endswith(".json")]
    all_frames = []

    for file in files:
        player_name = extract_player_name_from_filename(file)
        df = load_data(player_name)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)

def build_user_daily_summary(df):
    if df.empty:
        return pd.DataFrame()

    summary = df.groupby(["player_name", "practice_day"]).agg(
        total_shots=("result", "count"),
        makes=("result", lambda x: (x == "MAKE").sum()),
        misses=("result", lambda x: (x == "MISS").sum())
    ).reset_index()

    summary["FG%"] = (summary["makes"] / summary["total_shots"] * 100).round(1)

    return summary.rename(columns={
        "player_name": "User",
        "practice_day": "Practice Day",
        "total_shots": "Total Shots",
        "makes": "Makes",
        "misses": "Misses"
    })

def build_user_session_summary(df):
    if df.empty:
        return pd.DataFrame()

    summary = df.groupby(["player_name", "practice_day", "video_file"]).agg(
        total_shots=("result", "count"),
        makes=("result", lambda x: (x == "MAKE").sum()),
        misses=("result", lambda x: (x == "MISS").sum())
    ).reset_index()

    summary["FG%"] = (summary["makes"] / summary["total_shots"] * 100).round(1)

    return summary.rename(columns={
        "player_name": "User",
        "practice_day": "Practice Day",
        "video_file": "Session Name",
        "total_shots": "Total Shots",
        "makes": "Makes",
        "misses": "Misses"
    })

def get_hoop_roi():
    return tuple(int(v) for v in st.session_state.HOOP_ROI)

def get_net_roi():
    return tuple(int(v) for v in st.session_state.NET_ROI)

def nudge_roi(roi_name, dx=0, dy=0, dw=0, dh=0):
    if roi_name == "HOOP":
        x, y, w, h = st.session_state.HOOP_ROI
    else:
        x, y, w, h = st.session_state.NET_ROI

    x = max(0, min(959, x + dx))
    y = max(0, min(539, y + dy))
    w = max(10, min(400, w + dw))
    h = max(10, min(300, h + dh))

    if roi_name == "HOOP":
        st.session_state.HOOP_ROI = (x, y, w, h)
    else:
        st.session_state.NET_ROI = (x, y, w, h)

def draw_preview(frame, hoop_roi, net_roi):
    preview = frame.copy()

    hx, hy, hw, hh = hoop_roi
    nx, ny, nw, nh = net_roi

    cv2.rectangle(preview, (hx, hy), (hx + hw, hy + hh), (255, 0, 0), 2)
    cv2.putText(preview, "Hoop ROI", (hx, max(20, hy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.rectangle(preview, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)
    cv2.putText(preview, "Net ROI", (nx, max(20, ny - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return preview

def resize_video_to_1080p(input_path):
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 0:
        fps = 30.0

    if width <= 1920 and height <= 1080:
        cap.release()
        return input_path

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_out, fourcc, fps, (1920, 1080))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (1920, 1080))
        out.write(resized)

    cap.release()
    out.release()

    return temp_out

def get_video_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    return cv2.resize(frame, (960, 540))

def process_video_with_optional_roi(video_path, session_name, use_calibrated_roi):
    if not use_calibrated_roi:
        return process_video(video_path, session_name)

    try:
        sig = inspect.signature(process_video)
        kwargs = {}
        if "hoop_roi" in sig.parameters:
            kwargs["hoop_roi"] = get_hoop_roi()
        if "net_roi" in sig.parameters:
            kwargs["net_roi"] = get_net_roi()

        if kwargs:
            return process_video(video_path, session_name, **kwargs)

        return process_video(video_path, session_name)
    except TypeError:
        return process_video(video_path, session_name)

# ---------------- LIVE SETTINGS ----------------
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class LiveVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.prev_frame = None
        self.motion_frames = 0
        self.last_event_time = 0
        self.last_result = ""
        self.last_result_color = (255, 255, 255)
        self.result_hold_until = 0
        self.session_start_time = time.time()
        self.lock = threading.Lock()
        self.new_events = []

        self.hoop_roi = DEFAULT_HOOP_ROI
        self.net_roi = DEFAULT_NET_ROI
        self.detect_mode = False
        self.motion_thresh = 35
        self.net_thresh = 28
        self.min_area = 220
        self.motion_frames_needed = 3
        self.cooldown_seconds = 2.0
        self.result_hold_seconds = 1.2

    def update_settings(
        self,
        hoop_roi,
        net_roi,
        detect_mode,
        motion_thresh,
        net_thresh,
        min_area,
        motion_frames_needed,
        cooldown_seconds,
        result_hold_seconds,
    ):
        with self.lock:
            self.hoop_roi = tuple(int(v) for v in hoop_roi)
            self.net_roi = tuple(int(v) for v in net_roi)
            self.detect_mode = bool(detect_mode)
            self.motion_thresh = int(motion_thresh)
            self.net_thresh = int(net_thresh)
            self.min_area = int(min_area)
            self.motion_frames_needed = int(motion_frames_needed)
            self.cooldown_seconds = float(cooldown_seconds)
            self.result_hold_seconds = float(result_hold_seconds)

    def get_motion(self, frame_a, frame_b, rect, thresh):
        x, y, w, h = rect

        roi_a = frame_a[y:y+h, x:x+w]
        roi_b = frame_b[y:y+h, x:x+w]

        if roi_a.size == 0 or roi_b.size == 0:
            return []

        diff = cv2.absdiff(roi_a, roi_b)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh_img = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh_img, None, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (960, 540))

        with self.lock:
            hoop_roi = self.hoop_roi
            net_roi = self.net_roi
            detect_mode = self.detect_mode
            motion_thresh = self.motion_thresh
            net_thresh = self.net_thresh
            min_area = self.min_area
            motion_frames_needed = self.motion_frames_needed
            cooldown_seconds = self.cooldown_seconds
            result_hold_seconds = self.result_hold_seconds

        x, y, w, h = hoop_roi
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, "Hoop ROI", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        x2, y2, w2, h2 = net_roi
        cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
        cv2.putText(img, "Net ROI", (x2, max(20, y2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if not detect_mode:
            self.prev_frame = img.copy()
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        if self.prev_frame is None:
            self.prev_frame = img.copy()
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        hoop_contours = self.get_motion(self.prev_frame, img, hoop_roi, motion_thresh)
        large_hoop = [c for c in hoop_contours if cv2.contourArea(c) >= min_area]

        if len(large_hoop) > 0:
            self.motion_frames += 1
        else:
            self.motion_frames = 0

        current_time = time.time()

        if self.motion_frames >= motion_frames_needed and (current_time - self.last_event_time) > cooldown_seconds:
            net_contours = self.get_motion(self.prev_frame, img, net_roi, net_thresh)
            large_net = [c for c in net_contours if cv2.contourArea(c) >= min_area]

            if len(large_net) > 0:
                result = "MAKE"
                color = (0, 255, 0)
            else:
                result = "MISS"
                color = (0, 0, 255)

            self.last_result = f"{result} - Field Goal"
            self.last_result_color = color
            self.result_hold_until = current_time + result_hold_seconds
            self.last_event_time = current_time
            self.motion_frames = 0

            event = {
                "time_sec": round(current_time - self.session_start_time, 2),
                "result": result,
                "shot_zone": "Field Goal",
                "video_file": "Live Session",
                "created_at": datetime.now().isoformat(),
                "player_name": player
            }

            with self.lock:
                self.new_events.append(event)

        if current_time < self.result_hold_until:
            cv2.putText(
                img,
                self.last_result,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                self.last_result_color,
                3
            )

        self.prev_frame = img.copy()
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=140)

    st.title("ShotSense")
    player = st.text_input("Player Name", "Player 1")
    team = st.text_input("Team", "Training")

    if os.path.exists("headshot.png"):
        st.image("headshot.png", width=160, caption=player)

    st.markdown("---")
    st.subheader("Video Processing")

    session_name = st.text_input("Session Name", "Practice Day 1")
    uploaded_file = st.file_uploader("Upload Basketball Video", type=["mp4", "mov", "m4v", "avi"])
    use_upload_calibration = st.checkbox("Use current calibrated ROI for uploaded video", value=True)

    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")
        st.caption("Video uploaded and ready for processing.")
        if use_upload_calibration:
            st.caption("Uploaded video will use the same hoop and net ROI currently saved in the app.")
            st.caption(f"Upload Hoop ROI: {get_hoop_roi()}")
            st.caption(f"Upload Net ROI: {get_net_roi()}")

        # --------- ADDED UPLOAD PREVIEW ONLY ---------
        st.markdown("### Video Preview")
        st.video(uploaded_file)
        # ---------------------------------------------

    if st.button("Run Shot Detection"):
        if uploaded_file is None:
            st.error("Please upload a video first.")
        else:
            temp_path = None
            optimized_path = None
            try:
                ext = os.path.splitext(uploaded_file.name)[1].lower() or ".mp4"

                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_file.flush()
                    temp_path = tmp_file.name

                if not os.path.exists(temp_path):
                    st.error("File was not saved.")
                    st.stop()

                if os.path.getsize(temp_path) == 0:
                    st.error("Saved file is empty.")
                    st.stop()

                with st.spinner("Optimizing video (1080p)..."):
                    optimized_path = resize_video_to_1080p(temp_path)

                preview_frame = get_video_first_frame(optimized_path)
                if preview_frame is not None and use_upload_calibration:
                    st.image(
                        cv2.cvtColor(draw_preview(preview_frame, get_hoop_roi(), get_net_roi()), cv2.COLOR_BGR2RGB),
                        caption="Upload Preview with Current ROI",
                        use_container_width=True,
                    )

                with st.spinner("Processing video..."):
                    shot_events = process_video_with_optional_roi(
                        optimized_path,
                        session_name,
                        use_calibrated_roi=use_upload_calibration
                    )

                if shot_events is None:
                    st.error("Detection returned no result.")
                elif len(shot_events) == 0:
                    st.warning("Video processed but no shots were detected.")
                else:
                    save_data(player, shot_events)
                    st.session_state.cloud_events = []
                    st.success(f"Detected {len(shot_events)} shots for {player}!")
                    st.rerun()

            except Exception as e:
                st.error(f"Shot detection failed: {type(e).__name__}: {e}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

                if optimized_path and optimized_path != temp_path and os.path.exists(optimized_path):
                    os.remove(optimized_path)

    st.markdown("---")
    page = st.radio("Navigation", ["Profile", "Users", "Analytics", "Live"])

# ---------------- LOAD DATA ----------------
df = load_data(player)
all_users_df = load_all_users_data()

# ---------------- EMPTY STATE ----------------
if df.empty and page == "Profile":
    st.title("Player Profile")
    st.markdown(f"No saved data found yet for **{player}**. Run upload or live detection using this player name first.")
    st.stop()

# ---------------- OVERALL STATS ----------------
if not df.empty:
    total_shots = len(df)
    makes = len(df[df["result"] == "MAKE"])
    misses = len(df[df["result"] == "MISS"])
    fg = (makes / total_shots * 100) if total_shots > 0 else 0

    last_day = df["practice_day"].max()
    last_df = df[df["practice_day"] == last_day].copy()

    last_total = len(last_df)
    last_makes = len(last_df[last_df["result"] == "MAKE"])
    last_fg = (last_makes / last_total * 100) if last_total > 0 else 0

# ================= PROFILE PAGE =================
if page == "Profile":
    st.title("Player Profile")
    st.write(f"{player} • {team}")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Session Selector")
    session_options = sorted(df["practice_day"].unique(), reverse=True)
    selected_session = st.selectbox("Select Practice Day", session_options, index=0)
    session_df = df[df["practice_day"] == selected_session].copy()
    st.markdown("</div>", unsafe_allow_html=True)

    session_total = len(session_df)
    session_makes = len(session_df[session_df["result"] == "MAKE"])
    session_fg = (session_makes / session_total * 100) if session_total > 0 else 0

    st.markdown(f"""
    <div class="summary-box">
        Last Practice: {last_total} shots • {last_fg:.1f}% FG
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Recent Activity Feed")
        st.markdown(f'<div class="feed-item">Last session: {last_total} shots, {last_fg:.1f}% FG</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="feed-item">Selected session: {selected_session} — {session_total} shots, {session_fg:.1f}% FG</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Session Overview")
        st.markdown(f'<div class="feed-item">Shots: {session_total}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="feed-item">FG%: {session_fg:.1f}%</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric("Shots", total_shots)
    with c2:
        metric("Makes", makes)
    with c3:
        metric("Misses", misses)
    with c4:
        metric("FG%", f"{fg:.1f}%")

    col3, col4 = st.columns([1, 2])

    with col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("FG% Donut")

        donut_df = pd.DataFrame({
            "Category": ["Makes", "Misses"],
            "Value": [makes, misses]
        })

        donut = alt.Chart(donut_df).mark_arc(innerRadius=70).encode(
            theta="Value:Q",
            color=alt.Color(
                "Category:N",
                scale=alt.Scale(domain=["Makes", "Misses"], range=[GREEN, RED]),
                legend=None
            ),
            tooltip=["Category", "Value"]
        ).properties(height=280)

        center_text = alt.Chart(pd.DataFrame({"text": [f"{fg:.1f}% FG"]})).mark_text(
            fontSize=22,
            fontWeight="bold",
            color=WHITE
        ).encode(text="text:N")

        st.altair_chart(donut + center_text, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Shots by Practice Day")

        daily_trend = df.groupby("practice_day").agg(
            total_shots=("result", "count")
        ).reset_index()

        trend_chart = alt.Chart(daily_trend).mark_line(
            color=BLUE,
            point=alt.OverlayMarkDef(color=WHITE, size=70)
        ).encode(
            x=alt.X("practice_day:T", title="Practice Day"),
            y=alt.Y("total_shots:Q", title="Total Shots"),
            tooltip=["practice_day", "total_shots"]
        ).properties(height=280)

        st.altair_chart(trend_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    tabs = st.tabs(["Selected Session Log", "Daily Summary", "Monthly Summary"])

    with tabs[0]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Selected Session Shot Log")
        st.dataframe(clean_display_table(session_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        daily_summary = df.groupby("practice_day").agg(
            total_shots=("result", "count"),
            makes=("result", lambda x: (x == "MAKE").sum()),
            misses=("result", lambda x: (x == "MISS").sum())
        ).reset_index()

        daily_summary["Shooting %"] = (
            daily_summary["makes"] / daily_summary["total_shots"] * 100
        ).round(1)

        daily_summary = daily_summary.rename(columns={
            "practice_day": "Practice Day",
            "total_shots": "Total Shots",
            "makes": "Makes",
            "misses": "Misses"
        })

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Daily Session Summary")
        st.dataframe(daily_summary, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        monthly_summary = df.groupby("practice_month").agg(
            total_shots=("result", "count"),
            makes=("result", lambda x: (x == "MAKE").sum()),
            misses=("result", lambda x: (x == "MISS").sum()),
            session_count=("practice_day", "nunique")
        ).reset_index()

        monthly_summary["Shooting %"] = (
            monthly_summary["makes"] / monthly_summary["total_shots"] * 100
        ).round(1)

        monthly_summary = monthly_summary.rename(columns={
            "practice_month": "Practice Month",
            "total_shots": "Total Shots",
            "makes": "Makes",
            "misses": "Misses",
            "session_count": "Session Count"
        })

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Monthly Session Summary")
        st.dataframe(monthly_summary, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ================= USERS PAGE =================
elif page == "Users":
    st.title("All Users")

    if all_users_df.empty:
        st.info("No saved user data yet.")
    else:
        user_daily = build_user_daily_summary(all_users_df)
        user_sessions = build_user_session_summary(all_users_df)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("All Users Daily Stats")
        st.dataframe(user_daily, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("All Users Session Stats")
        st.dataframe(user_sessions, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("All Saved Shot Logs")
        st.dataframe(clean_display_table(all_users_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ================= ANALYTICS PAGE =================
elif page == "Analytics":
    st.title("Analytics")

    if df.empty:
        st.info("No saved data found for this player yet.")
    else:
        zone_counts = df.groupby(["shot_zone", "result"]).size().reset_index(name="count")

        chart = alt.Chart(zone_counts).mark_bar().encode(
            x=alt.X("shot_zone:N", title="Shot Zone"),
            y=alt.Y("count:Q", title="Number of Shots"),
            color=alt.Color(
                "result:N",
                scale=alt.Scale(domain=["MAKE", "MISS"], range=[GREEN, RED]),
                title="Result"
            ),
            xOffset="result:N",
            tooltip=["shot_zone", "result", "count"]
        ).properties(height=360)

        st.altair_chart(chart, use_container_width=True)

# ================= LIVE PAGE =================
elif page == "Live":
    st.title("Live Shot Detection")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Setup Instructions")
    st.write("1. Place camera in a fixed position.")
    st.write("2. Use Setup Mode to line up the existing hoop and net boxes.")
    st.write("3. Adjust sensitivity settings if needed.")
    st.write("4. Save calibration.")
    st.write("5. Start live detection.")

    st.warning("Keep the camera fixed after setup. If the camera moves, recapture a calibration frame and adjust the ROI again.")

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Hoop ROI:** {get_hoop_roi()}")
    with c2:
        st.write(f"**Net ROI:** {get_net_roi()}")

    m1, m2, m3 = st.columns(3)
    with m1:
        if st.button("Setup Mode"):
            st.session_state.detect_mode = False
    with m2:
        if st.button("Start Detection"):
            st.session_state.detect_mode = True
    with m3:
        if st.button("Stop Detection"):
            st.session_state.detect_mode = False

    if st.session_state.detect_mode:
        st.success("Detection Mode: ON")
    else:
        st.info("Setup Mode: ON")

    st.markdown("### Calibration Capture")
    st.caption("Use this camera capture to line up and save the hoop and net ROI before starting continuous live detection.")

    calibration_image = st.camera_input("Calibration Camera", key="calibration_camera_input")
    if calibration_image is not None:
        st.session_state.calib_frame_bytes = calibration_image.getvalue()

    calib_frame = None
    if st.session_state.calib_frame_bytes is not None:
        file_bytes = np.asarray(bytearray(st.session_state.calib_frame_bytes), dtype=np.uint8)
        calib_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if calib_frame is not None:
        calib_frame = cv2.resize(calib_frame, (960, 540))
        calib_preview = draw_preview(calib_frame, get_hoop_roi(), get_net_roi())
        st.image(
            cv2.cvtColor(calib_preview, cv2.COLOR_BGR2RGB),
            caption="Calibration Preview",
            use_container_width=True,
        )

    save1, save2 = st.columns(2)
    with save1:
        if st.button("Save Live Calibration"):
            st.success("Calibration saved.")
    with save2:
        if st.button("Clear Calibration Frame"):
            st.session_state.calib_frame_bytes = None
            st.success("Calibration frame cleared.")

    st.markdown("### Continuous Live Detection")
    st.caption("Start the live camera below after calibration is saved. This stream performs continuous MAKE/MISS detection using the saved ROI.")

    ctx = webrtc_streamer(
        key="shotsense-live-detect",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=LiveVideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30}
            },
            "audio": False
        },
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.update_settings(
            hoop_roi=get_hoop_roi(),
            net_roi=get_net_roi(),
            detect_mode=st.session_state.detect_mode,
            motion_thresh=st.session_state.motion_thresh,
            net_thresh=st.session_state.net_thresh,
            min_area=st.session_state.min_area,
            motion_frames_needed=st.session_state.motion_frames_needed,
            cooldown_seconds=st.session_state.cooldown_seconds,
            result_hold_seconds=st.session_state.result_hold_seconds,
        )

    st.write(f"Current session shots collected: **{len(st.session_state.cloud_events)}**")

    sync1, sync2 = st.columns(2)
    with sync1:
        if st.button("Sync Live Results"):
            if ctx.state.playing and ctx.video_processor:
                new_events = []
                with ctx.video_processor.lock:
                    if ctx.video_processor.new_events:
                        new_events = ctx.video_processor.new_events.copy()
                        ctx.video_processor.new_events.clear()

                if new_events:
                    save_data(player, new_events)
                    st.session_state.cloud_events = []
                    st.success(f"Synced {len(new_events)} live shot(s).")
                    st.rerun()
                else:
                    st.info("No new live shots to sync yet.")
            else:
                st.warning("Start the live camera first.")

    with sync2:
        if st.button("Clear Current Session Shots"):
            st.session_state.cloud_events = []
            st.success("Current session shots cleared.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- CALIBRATION TOOLS ----------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Calibration Tools")
    st.write("Move the existing boxes while watching the calibration preview. Your original ROI values are kept as the default.")

    st.markdown("### Hoop Box Controls")
    hc1, hc2, hc3, hc4 = st.columns(4)

    with hc1:
        if st.button("Hoop ←"):
            nudge_roi("HOOP", dx=-5)
        if st.button("Hoop ↑"):
            nudge_roi("HOOP", dy=-5)

    with hc2:
        if st.button("Hoop →"):
            nudge_roi("HOOP", dx=5)
        if st.button("Hoop ↓"):
            nudge_roi("HOOP", dy=5)

    with hc3:
        if st.button("Hoop Wider"):
            nudge_roi("HOOP", dw=5)
        if st.button("Hoop Taller"):
            nudge_roi("HOOP", dh=5)

    with hc4:
        if st.button("Hoop Narrower"):
            nudge_roi("HOOP", dw=-5)
        if st.button("Hoop Shorter"):
            nudge_roi("HOOP", dh=-5)

    st.markdown("### Net Box Controls")
    nc1, nc2, nc3, nc4 = st.columns(4)

    with nc1:
        if st.button("Net ←"):
            nudge_roi("NET", dx=-5)
        if st.button("Net ↑"):
            nudge_roi("NET", dy=-5)

    with nc2:
        if st.button("Net →"):
            nudge_roi("NET", dx=5)
        if st.button("Net ↓"):
            nudge_roi("NET", dy=5)

    with nc3:
        if st.button("Net Wider"):
            nudge_roi("NET", dw=5)
        if st.button("Net Taller"):
            nudge_roi("NET", dh=5)

    with nc4:
        if st.button("Net Narrower"):
            nudge_roi("NET", dw=-5)
        if st.button("Net Shorter"):
            nudge_roi("NET", dh=-5)

    st.markdown("### Sensitivity Settings")
    st.caption("Motion Threshold: how much movement near the rim counts as a shot.")
    st.caption("Net Threshold: how much the net has to move for it to count as a make.")
    st.caption("Minimum Contour Area: ignores small movements that aren’t real shots.")
    st.caption("Motion Frames Required: makes sure the ball is seen long enough before counting it.")
    st.caption("Cooldown Seconds: stops one shot from being counted multiple times.")
    st.caption("Result Display Time: how long MAKE or MISS stays on screen.")

    s1, s2 = st.columns(2)

    with s1:
        st.session_state.motion_thresh = st.slider("Motion Threshold", 10, 80, st.session_state.motion_thresh)
        st.session_state.net_thresh = st.slider("Net Threshold", 10, 80, st.session_state.net_thresh)
        st.session_state.min_area = st.slider("Minimum Contour Area", 50, 2000, st.session_state.min_area)

    with s2:
        st.session_state.motion_frames_needed = st.slider("Motion Frames Required", 1, 8, st.session_state.motion_frames_needed)
        st.session_state.cooldown_seconds = st.slider("Cooldown Seconds", 1.0, 5.0, st.session_state.cooldown_seconds)
        st.session_state.result_hold_seconds = st.slider("Result Display Time", 0.5, 3.0, st.session_state.result_hold_seconds)

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Save Live Calibration 2"):
            st.success("Calibration saved.")
    with b2:
        if st.button("Reset to Default ROI"):
            st.session_state.HOOP_ROI = DEFAULT_HOOP_ROI
            st.session_state.NET_ROI = DEFAULT_NET_ROI
            st.session_state.motion_thresh = 35
            st.session_state.net_thresh = 28
            st.session_state.min_area = 220
            st.session_state.motion_frames_needed = 3
            st.session_state.cooldown_seconds = 2.0
            st.session_state.result_hold_seconds = 1.2
            st.session_state.detect_mode = False
            st.session_state.calib_frame_bytes = None
            st.success("ROIs reset to original defaults.")

    st.write(f"**Current Hoop ROI:** {get_hoop_roi()}")
    st.write(f"**Current Net ROI:** {get_net_roi()}")

    st.markdown("</div>", unsafe_allow_html=True)
