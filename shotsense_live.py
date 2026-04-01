import cv2
import time

# ---------------- OPEN WEBCAM ----------------
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Could not open webcam.")
    raise SystemExit(1)

# Try a larger webcam size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

time.sleep(2.0)

def get_valid_frame(cap, tries=20):
    for _ in range(tries):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return True, frame
        time.sleep(0.1)
    return False, None

ret, frame1 = get_valid_frame(cap)
ret2, frame2 = get_valid_frame(cap)

if not ret or not ret2 or frame1 is None or frame2 is None:
    print("Could not start webcam frames.")
    cap.release()
    raise SystemExit(1)

print("Live frame shape:", frame1.shape)

# ---------------- ROI ----------------
# These may still need adjustment for live webcam framing
FULL_ROI_X, FULL_ROI_Y, FULL_ROI_W, FULL_ROI_H = 1081, 791, 206, 74
FULL_NET_X, FULL_NET_Y, FULL_NET_W, FULL_NET_H = 1127, 901, 97, 83

PROCESS_SCALE = 0.5

# ---------------- SETTINGS ----------------
HOOP_THRESH = 35
HOOP_MIN_AREA = 140
HOOP_TRIGGER_FRAMES = 1

NET_THRESH = 28
NET_MIN_AREA = 220
NET_HIT_FRAMES_REQUIRED = 3
NET_STRONG_MAKE_AREA = 420

CLASSIFY_DELAY_SECONDS = 0.35
CLASSIFY_WINDOW_SECONDS = 1.6
COOLDOWN_SECONDS = 4

# ---------------- VARIABLES ----------------
motion_frames = 0
last_event_time = 0

last_result_text = ""
last_result_color = (255, 255, 255)
result_display_frames = 0
RESULT_HOLD_FRAMES = 35

# ---------------- SCALE ROI ----------------
ROI_X = int(FULL_ROI_X * PROCESS_SCALE)
ROI_Y = int(FULL_ROI_Y * PROCESS_SCALE)
ROI_W = int(FULL_ROI_W * PROCESS_SCALE)
ROI_H = int(FULL_ROI_H * PROCESS_SCALE)

NET_X = int(FULL_NET_X * PROCESS_SCALE)
NET_Y = int(FULL_NET_Y * PROCESS_SCALE)
NET_W = int(FULL_NET_W * PROCESS_SCALE)
NET_H = int(FULL_NET_H * PROCESS_SCALE)

HOOP_CENTER_X = ROI_X + ROI_W // 2

def get_shot_zone(x, y):
    distance = abs(x - HOOP_CENTER_X)
    if distance < 90:
        return "Paint"
    return "Mid-Range"

def roi_is_valid(frame, rect):
    x, y, w, h = rect
    fh, fw = frame.shape[:2]
    return x >= 0 and y >= 0 and w > 0 and h > 0 and (x + w) <= fw and (y + h) <= fh

def get_motion_contours(frame_a, frame_b, rect, thresh_value):
    x, y, w, h = rect

    # Prevent empty ROI crashes
    if not roi_is_valid(frame_a, rect) or not roi_is_valid(frame_b, rect):
        return []

    roi_a = frame_a[y:y+h, x:x+w]
    roi_b = frame_b[y:y+h, x:x+w]

    if roi_a.size == 0 or roi_b.size == 0:
        return []

    diff = cv2.absdiff(roi_a, roi_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh_img = cv2.threshold(blur, thresh_value, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh_img, None, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def analyze_make_miss_live(cap, current_frame):
    start_time = time.time()
    net_hit_frames = 0
    max_net_area = 0
    prev_local = current_frame.copy()
    last_good_frame = current_frame.copy()

    time.sleep(CLASSIFY_DELAY_SECONDS)

    while time.time() - start_time < CLASSIFY_WINDOW_SECONDS:
        ret_next, next_frame = cap.read()
        if not ret_next or next_frame is None:
            break

        if PROCESS_SCALE != 1.0:
            next_frame = cv2.resize(next_frame, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE)

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

    make_detected = (
        net_hit_frames >= NET_HIT_FRAMES_REQUIRED
        or max_net_area >= NET_STRONG_MAKE_AREA
    )

    return make_detected, last_good_frame, max_net_area, net_hit_frames

# ---------------- SCALE FIRST FRAMES ----------------
if PROCESS_SCALE != 1.0:
    frame1 = cv2.resize(frame1, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE)
    frame2 = cv2.resize(frame2, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE)

print("Scaled frame shape:", frame1.shape)
print("Hoop ROI:", (ROI_X, ROI_Y, ROI_W, ROI_H))
print("Net ROI:", (NET_X, NET_Y, NET_W, NET_H))

# ---------------- MAIN LOOP ----------------
while cap.isOpened():

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

    display_frame = frame1.copy()

    # Only draw if valid
    if roi_is_valid(display_frame, (ROI_X, ROI_Y, ROI_W, ROI_H)):
        cv2.rectangle(display_frame, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (255, 0, 0), 2)

    if roi_is_valid(display_frame, (NET_X, NET_Y, NET_W, NET_H)):
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

    if (
        motion_frames >= HOOP_TRIGGER_FRAMES
        and (time.time() - last_event_time) > COOLDOWN_SECONDS
        and len(large_hoop_contours) > 0
    ):
        largest_contour = max(large_hoop_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        contour_center_x = ROI_X + x + w // 2
        contour_center_y = ROI_Y + y + h // 2

        shot_zone = get_shot_zone(contour_center_x, contour_center_y)

        make_detected, frame2_after, max_net_area, net_hit_frames = analyze_make_miss_live(cap, frame2)

        result = "MAKE" if make_detected else "MISS"

        last_result_text = f"{result} - {shot_zone}"
        last_result_color = (0, 255, 0) if result == "MAKE" else (0, 0, 255)
        result_display_frames = RESULT_HOLD_FRAMES

        print(f"LIVE SHOT -> {result}")

        last_event_time = time.time()
        motion_frames = 0
        frame1 = frame2_after

        ret_next_main, frame2 = cap.read()
        if not ret_next_main or frame2 is None:
            break

        if PROCESS_SCALE != 1.0:
            frame2 = cv2.resize(frame2, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE)

    else:
        frame1 = frame2
        ret_next_main, frame2 = cap.read()
        if not ret_next_main or frame2 is None:
            break

        if PROCESS_SCALE != 1.0:
            frame2 = cv2.resize(frame2, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE)

    cv2.imshow("ShotSense Live", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
