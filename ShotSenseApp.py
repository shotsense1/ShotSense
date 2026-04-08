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

# ---------------- SESSION STATE ----------------
if "cloud_events" not in st.session_state:
    st.session_state.cloud_events = []

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

def clean_display_table(source_df):
    display_df = source_df.copy()
    for col in ["hoop_roi", "net_roi", "shot_location"]:
        if col in display_df.columns:
            display_df = display_df.drop(columns=[col])

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

# ---------------- LIVE SETTINGS ----------------
HOOP_ROI = (540, 395, 103, 37)
NET_ROI = (563, 450, 48, 41)

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

    def get_motion(self, frame_a, frame_b, rect, thresh):
        x, y, w, h = rect

        roi_a = frame_a[y:y+h, x:x+w]
        roi_b = frame_b[y:y+h, x:x+w]

        diff = cv2.absdiff(roi_a, roi_b)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh_img = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh_img, None, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def recv(self, frame):
        import time

        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (960, 540))

        x, y, w, h = HOOP_ROI
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, "Hoop ROI", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        x2, y2, w2, h2 = NET_ROI
        cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
        cv2.putText(img, "Net ROI", (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.prev_frame is None:
            self.prev_frame = img.copy()
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        hoop_contours = self.get_motion(self.prev_frame, img, HOOP_ROI, 35)
        large_hoop = [c for c in hoop_contours if cv2.contourArea(c) >= 220]

        if len(large_hoop) > 0:
            self.motion_frames += 1
        else:
            self.motion_frames = 0

        current_time = time.time()

        if self.motion_frames >= 3 and (current_time - self.last_event_time) > 2.0:
            net_contours = self.get_motion(self.prev_frame, img, NET_ROI, 28)
            large_net = [c for c in net_contours if cv2.contourArea(c) >= 220]

            if len(large_net) > 0:
                result = "MAKE"
                color = (0, 255, 0)
            else:
                result = "MISS"
                color = (0, 0, 255)

            self.last_result = f"{result} - Field Goal"
            self.last_result_color = color
            self.result_hold_until = current_time + 1.2
            self.last_event_time = current_time
            self.motion_frames = 0

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

# ---------------- LOAD DATA ----------------
df = load_data()

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

    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")
        st.video(uploaded_file)

    if st.button("Run Shot Detection"):
        if uploaded_file is None:
            st.error("Please upload a video first.")
        else:
            temp_path = None
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

                with st.spinner("Processing video..."):
                    shot_events = process_video(temp_path, session_name)

                if shot_events is None:
                    st.error("Detection returned no result.")
                elif len(shot_events) == 0:
                    st.warning("Video processed but no shots were detected.")
                else:
                    st.session_state.cloud_events.extend(shot_events)
                    st.success(f"Detected {len(shot_events)} shots!")
                    st.rerun()

            except Exception as e:
                st.error(f"Shot detection failed: {type(e).__name__}: {e}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    st.markdown("---")
    page = st.radio("Navigation", ["Profile", "Analytics", "Live"])

# ---------------- EMPTY STATE ----------------
if df.empty and page != "Live":
    st.title("ShotSense Dashboard")
    st.markdown("Upload a video, enter a session name, and click **Run Shot Detection** to generate shot data.")
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

        daily_summary["shooting_percentage"] = (
            daily_summary["makes"] / daily_summary["total_shots"] * 100
        ).round(1)

        daily_summary = daily_summary.rename(columns={
            "practice_day": "Practice Day",
            "total_shots": "Total Shots",
            "makes": "Makes",
            "misses": "Misses",
            "shooting_percentage": "Shooting %"
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

        monthly_summary["shooting_percentage"] = (
            monthly_summary["makes"] / monthly_summary["total_shots"] * 100
        ).round(1)

        monthly_summary = monthly_summary.rename(columns={
            "practice_month": "Practice Month",
            "total_shots": "Total Shots",
            "makes": "Makes",
            "misses": "Misses",
            "session_count": "Session Count",
            "shooting_percentage": "Shooting %"
        })

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Monthly Session Summary")
        st.dataframe(monthly_summary, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ================= ANALYTICS PAGE =================
elif page == "Analytics":
    st.title("Analytics")

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
    st.subheader("Browser Live View")
    st.write("Live preview can work in the deployed version, but uploaded video processing is the reliable cloud-safe path.")

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Hoop ROI:** {HOOP_ROI}")
    with c2:
        st.write(f"**Net ROI:** {NET_ROI}")

    webrtc_streamer(
        key="shotsense-live-cloud",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=LiveVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)
