import cv2
import json
import os
from datetime import datetime

# ---------------- VIDEO FILE ----------------
VIDEO = "uploaded_video.mp4"

# ---------------- SESSION NAME ----------------
session_name = VIDEO
if os.path.exists("current_session_name.txt"):
    with open("current_session_name.txt", "r") as f:
        session_name = f.read().strip()

# ---------------- FULL-RES ROI VALUES ----------------
# Blue box = shot trigger only
FULL_ROI_X, FULL_ROI_Y, FULL_ROI_W, FULL_ROI_H = 1050, 760, 270, 130

# Green box = make detection only
FULL_NET_X, FULL_NET_Y, FULL_NET_W, FULL_NET_H = 1127, 901, 97, 83

# ---------------- PROCESSING SCALE ----------------
PROCESS_SCALE = 0.5

# ---------------- BLUE BOX SETTINGS ----------------
HOOP_THRESH = 35
HOOP_MIN_AREA = 160
HOOP_TRIGGER_FRAMES = 2

# ---------------- GREEN BOX SETTINGS ----------------
NET_THRESH = 28
NET_MIN_AREA = 220
NET_HIT_FRAMES_REQUIRED = 3
NET_STRONG_MAKE_AREA = 420
CLASSIFY_DELAY_FRAMES = 10
CLASSIFY_WINDOW_FRAMES = 85

# ---------------- COOLDOWN ----------------
COOLDOWN_FRAMES = 120

# ---------------- VARIABLES ----------------
shot_events = []
last_event_frame = -999999
motion_frames = 0

# ---------------- RESULT DISPLAY ----------------
last_result_text = ""
last_result_color = (255, 255, 255)
result_display_frames = 0
RESULT_HOLD_FRAMES = 35

# ---------------- SCALE ROI VALUES ----------------
ROI_X = int(FULL_ROI_X * PROCESS_SCALE)
ROI_Y = int(FULL_ROI_Y * PROCESS_SCALE)
ROI_W = int(FULL_ROI_W * PROCESS_SCALE)
ROI_H = int(FULL_ROI_H * PROCESS_SCALE)

NET_X = int(FULL_NET_X * PROCESS_SCALE)
NET_Y = int(FULL_NET_Y * PROCESS_SCALE)
NET_W = int(FULL_NET_W * PROCESS_SCALE)
NET_H = int(FULL_NET_H * PROCESS_SCALE)

# ---------------- HOOP CENTER ----------------
HOOP_CENTER_X = ROI_X + ROI_W // 2
HOOP_CENTER_Y = ROI_Y + ROI_H // 2

# ---------------- SHOT ZONE FUNCTION ----------------
def get_shot_zone(x, y):
    distance = abs(x - HOOP_CENTER_X)
    if distance < 90:
        return "Paint"
    return "Mid-Range"

# ---------------- MOTION HELPER ----------------
def get_motion_contours(frame_a, frame_b, rect, thresh_value):
    x, y, w, h = rect

    roi_a = frame_a[y:y+h, x:x+w]
    roi_b = frame_b[y:y+h, x:x+w]

    diff = cv2.absdiff(roi_a, roi_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh_img = cv2.threshold(blur, thresh_value, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh_img, None, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# ---------------- GREEN BOX ONLY MAKE CHECK ----------------
def analyze_make_miss(cap, current_frame):
    frames_read = 0
    net_hit_frames = 0
    max_net_area = 0
    prev_local = current_frame.copy()
    last_good_frame = current_frame.copy()

    while frames_read < CLASSIFY_WINDOW_FRAMES:
        ret_next, next_frame = cap.read()
        if not ret_next or next_frame is None:
            break

        if PROCESS_SCALE != 1.0:
            next_frame = cv2.resize(next_frame, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE)

        if frames_read >= CLASSIFY_DELAY_FRAMES:
            net_contours = get_motion_contours(
                prev_local,
                next_frame,
                (NET_X, NET_Y, NET_W, NET_H),
                NET_THRESH
            )

            large_net_contours = [c for c in net_contours if cv2.contourArea(c) >= NET_MIN_AREA]

            if large_net_contours:
                largest_area = max(cv2.contourArea(c) for c in large_net_contours)
                max_net_area = max(max_net_area, largest_area)
                net_hit_frames += 1

        prev_local = next_frame.copy()
        last_good_frame = next_frame.copy()
        frames_read += 1

    make_detected = (
        net_hit_frames >= NET_HIT_FRAMES_REQUIRED
        or max_net_area >= NET_STRONG_MAKE_AREA
    )

    return make_detected, last_good_frame, max_net_area, net_hit_frames

# ---------------- OPEN VIDEO ----------------
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

ret, frame1 = cap.read()
ret2, frame2 = cap.read()

if not ret or not ret2 or frame1 is None or frame2 is None:
    print("Could not open uploaded_video.mp4 or video has too few frames.")
    cap.release()
    raise SystemExit(1)

if PROCESS_SCALE != 1.0:
    frame1 = cv2.resize(frame1, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE)
    frame2 = cv2.resize(frame2, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE)

frame_index = 1

while cap.isOpened():

    # ---------------- BLUE BOX = SHOT TRIGGER ONLY ----------------
    hoop_contours = get_motion_contours(
        frame1,
        frame2,
        (ROI_X, ROI_Y, ROI_W, ROI_H),
        HOOP_THRESH
    )

    large_hoop_contours = [c for c in hoop_contours if cv2.contourArea(c) >= HOOP_MIN_AREA]
    motion_in_hoop = len(large_hoop_contours) > 0

    if motion_in_hoop:
        motion_frames += 1
    else:
        motion_frames = 0

    # ---------------- DISPLAY ----------------
    display_frame = frame1.copy()
    cv2.rectangle(display_frame, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (255, 0, 0), 2)
    cv2.rectangle(display_frame, (NET_X, NET_Y), (NET_X + NET_W, NET_Y + NET_H), (0, 255, 0), 2)

    if result_display_frames > 0:
        cv2.putText(
            display_frame,
            last_result_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            last_result_color,
            2
        )
        result_display_frames -= 1

    # ---------------- SHOT DETECTION ----------------
    if (
        motion_frames >= HOOP_TRIGGER_FRAMES
        and (frame_index - last_event_frame) > COOLDOWN_FRAMES
        and len(large_hoop_contours) > 0
    ):
        t_sec = round(frame_index / fps, 2)

        # temporary safer location:
        # do not use hoop contour center as fake release point
        shot_location = (HOOP_CENTER_X, HOOP_CENTER_Y)
        shot_zone = "Field Goal"

        # ---------------- ONLY GREEN BOX DECIDES MAKE/MISS ----------------
        make_detected, frame2_after, max_net_area, net_hit_frames = analyze_make_miss(cap, frame2)

        if make_detected:
            result = "MAKE"
        else:
            result = "MISS"

        last_result_text = f"{result} - {shot_zone}"
        last_result_color = (0, 255, 0) if result == "MAKE" else (0, 0, 255)
        result_display_frames = RESULT_HOLD_FRAMES

        print(
            f"SHOT EVENT at {t_sec}s -> {result} ({shot_zone}) | "
            f"green_box_max_area={max_net_area:.1f}, green_box_hit_frames={net_hit_frames}"
        )

        shot_events.append({
            "time_sec": t_sec,
            "result": result,
            "shot_location": shot_location,
            "shot_zone": shot_zone,
            "video_file": session_name,
            "hoop_roi": [FULL_ROI_X, FULL_ROI_Y, FULL_ROI_W, FULL_ROI_H],
            "net_roi": [FULL_NET_X, FULL_NET_Y, FULL_NET_W, FULL_NET_H],
            "green_box_max_area": float(max_net_area),
            "green_box_hit_frames": int(net_hit_frames),
            "created_at": datetime.now().isoformat()
        })

        last_event_frame = frame_index
        motion_frames = 0

        frame1 = frame2_after

        ret_next_main, frame2 = cap.read()
        if not ret_next_main or frame2 is None:
            break

        if PROCESS_SCALE != 1.0:
            frame2 = cv2.resize(frame2, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE)

        frame_index += CLASSIFY_WINDOW_FRAMES + 1

    else:
        frame1 = frame2
        ret_next_main, frame2 = cap.read()
        if not ret_next_main or frame2 is None:
            break

        if PROCESS_SCALE != 1.0:
            frame2 = cv2.resize(frame2, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE)

        frame_index += 1

    cv2.imshow("ShotSense Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

output_file = "shot_events.json"
with open(output_file, "w") as f:
    json.dump(shot_events, f, indent=2)

print(f"\nSaved {len(shot_events)} shot events to {output_file}")
