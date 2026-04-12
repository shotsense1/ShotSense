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
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
from detector_cloud import process_video

st.set_page_config(page_title="ShotSense Dashboard", layout="wide")

# ---------------- SESSION STATE ----------------
if "cloud_events" not in st.session_state:
    st.session_state.cloud_events = []

# ---------------- CALIBRATION STATE ----------------
if "HOOP_ROI" not in st.session_state:
    st.session_state.HOOP_ROI = (540, 395, 103, 37)

if "NET_ROI" not in st.session_state:
    st.session_state.NET_ROI = (563, 450, 48, 41)

HOOP_ROI = st.session_state.HOOP_ROI
NET_ROI = st.session_state.NET_ROI

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
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (960, 540))

        # USE UPDATED ROIs
        x, y, w, h = st.session_state.HOOP_ROI
        x2, y2, w2, h2 = st.session_state.NET_ROI

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(img, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)

        if self.prev_frame is None:
            self.prev_frame = img.copy()
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        hoop_contours = self.get_motion(self.prev_frame, img, st.session_state.HOOP_ROI, 35)
        large_hoop = [c for c in hoop_contours if cv2.contourArea(c) >= 220]

        if len(large_hoop) > 0:
            self.motion_frames += 1
        else:
            self.motion_frames = 0

        current_time = time.time()

        if self.motion_frames >= 3 and (current_time - self.last_event_time) > 2.0:
            net_contours = self.get_motion(self.prev_frame, img, st.session_state.NET_ROI, 28)
            large_net = [c for c in net_contours if cv2.contourArea(c) >= 220]

            result = "MAKE" if len(large_net) > 0 else "MISS"

            self.last_event_time = current_time
            self.motion_frames = 0

            event = {
                "time_sec": round(current_time - self.session_start_time, 2),
                "result": result,
                "shot_zone": "Field Goal",
                "video_file": "Live Session",
                "created_at": datetime.now().isoformat()
            }

            with self.lock:
                self.new_events.append(event)

        self.prev_frame = img.copy()
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("ShotSense")

    st.subheader("Calibration Setup")

    col1, col2 = st.columns(2)

    with col1:
        hoop_x = st.number_input("Hoop X", 0, 960, HOOP_ROI[0])
        hoop_y = st.number_input("Hoop Y", 0, 540, HOOP_ROI[1])
        hoop_w = st.number_input("Hoop Width", 10, 500, HOOP_ROI[2])
        hoop_h = st.number_input("Hoop Height", 10, 500, HOOP_ROI[3])

    with col2:
        net_x = st.number_input("Net X", 0, 960, NET_ROI[0])
        net_y = st.number_input("Net Y", 0, 540, NET_ROI[1])
        net_w = st.number_input("Net Width", 10, 500, NET_ROI[2])
        net_h = st.number_input("Net Height", 10, 500, NET_ROI[3])

    if st.button("Save Calibration"):
        st.session_state.HOOP_ROI = (hoop_x, hoop_y, hoop_w, hoop_h)
        st.session_state.NET_ROI = (net_x, net_y, net_w, net_h)
        st.success("Calibration Saved!")

    st.markdown("---")
    page = st.radio("Navigation", ["Profile", "Analytics", "Live"])

# ---------------- LIVE PAGE ----------------
if page == "Live":
    st.title("Live Shot Detection")

    st.write(f"Hoop ROI: {st.session_state.HOOP_ROI}")
    st.write(f"Net ROI: {st.session_state.NET_ROI}")

    ctx = webrtc_streamer(
        key="live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=LiveVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    if ctx.state.playing and ctx.video_processor:
        with ctx.video_processor.lock:
            if ctx.video_processor.new_events:
                st.session_state.cloud_events.extend(ctx.video_processor.new_events)
                ctx.video_processor.new_events.clear()
