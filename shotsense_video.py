import cv2
import json
from datetime import datetime
from shotsense_core import detect_hoop_motion, detect_net_motion

def run_video_analysis(video_path, output_json="shot_events.json"):
    # -------- ROI --------
    ROI_X, ROI_Y, ROI_W, ROI_H = 1085, 775, 205, 125
    NET_X, NET_Y, NET_W, NET_H = 1125, 845, 105, 155

    HOOP_RECT = (ROI_X, ROI_Y, ROI_W, ROI_H)
    NET_RECT = (NET_X, NET_Y, NET_W, NET_H)

    # -------- SETTINGS --------
    HOOP_THRESH = 35
    HOOP_MIN_AREA = 140
    HOOP_TRIGGER_FRAMES = 1

    NET_THRESH = 28
    NET_MIN_AREA = 220
    NET_HIT_FRAMES_REQUIRED = 2
    NET_STRONG_MAKE_AREA = 360

    CLASSIFY_DELAY_FRAMES = 10
    CLASSIFY_WINDOW_FRAMES = 95
    COOLDOWN_FRAMES = 100

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    ret, frame1 = cap.read()
    ret2, frame2 = cap.read()

    if not ret or not ret2 or frame1 is None or frame2 is None:
        cap.release()
        raise ValueError("Could not open video or video has too few frames.")

    shot_events = []
    frame_index = 1
    last_event_frame = -999999
    motion_frames = 0

    while cap.isOpened():
        motion, hoop_contours = detect_hoop_motion(
            frame1, frame2, HOOP_RECT, HOOP_THRESH, HOOP_MIN_AREA
        )

        if motion:
            motion_frames += 1
        else:
            motion_frames = 0

        display_frame = frame1.copy()
        cv2.rectangle(display_frame, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (255, 0, 0), 2)
        cv2.rectangle(display_frame, (NET_X, NET_Y), (NET_X + NET_W, NET_Y + NET_H), (0, 255, 0), 2)

        if (
            motion_frames >= HOOP_TRIGGER_FRAMES
            and (frame_index - last_event_frame) > COOLDOWN_FRAMES
            and len(hoop_contours) > 0
        ):
            t_sec = round(frame_index / fps, 2)

            prev_local = frame2.copy()
            frames_read = 0
            net_hit_frames = 0
            max_net_area = 0
            last_good_frame = frame2.copy()

            while frames_read < CLASSIFY_WINDOW_FRAMES:
                ret_next, next_frame = cap.read()
                if not ret_next or next_frame is None:
                    break

                if frames_read >= CLASSIFY_DELAY_FRAMES:
                    net_motion, max_area, _ = detect_net_motion(
                        prev_local, next_frame, NET_RECT, NET_THRESH, NET_MIN_AREA
                    )
                    if net_motion:
                        net_hit_frames += 1
                        max_net_area = max(max_net_area, max_area)

                prev_local = next_frame.copy()
                last_good_frame = next_frame.copy()
                frames_read += 1

            make_detected = (
                net_hit_frames >= NET_HIT_FRAMES_REQUIRED
                or max_net_area >= NET_STRONG_MAKE_AREA
            )

            result = "MAKE" if make_detected else "MISS"

            shot_events.append({
                "time_sec": t_sec,
                "result": result,
                "green_box_max_area": float(max_net_area),
                "green_box_hit_frames": int(net_hit_frames),
                "created_at": datetime.now().isoformat()
            })

            print(f"SHOT EVENT at {t_sec}s -> {result}")

            last_event_frame = frame_index
            motion_frames = 0
            frame1 = last_good_frame

            ret_next_main, frame2 = cap.read()
            if not ret_next_main or frame2 is None:
                break

            frame_index += frames_read + 1

        else:
            frame1 = frame2
            ret_next_main, frame2 = cap.read()
            if not ret_next_main or frame2 is None:
                break
            frame_index += 1

        cv2.imshow("ShotSense Video", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    with open(output_json, "w") as f:
        json.dump(shot_events, f, indent=2)

    return shot_events

if __name__ == "__main__":
    events = run_video_analysis("uploaded_video.mp4")
    print(f"Saved {len(events)} shot events.")
