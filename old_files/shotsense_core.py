import cv2

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

def detect_hoop_motion(frame1, frame2, hoop_rect, hoop_thresh, hoop_min_area):
    contours = get_motion_contours(frame1, frame2, hoop_rect, hoop_thresh)
    large_contours = [c for c in contours if cv2.contourArea(c) >= hoop_min_area]
    return len(large_contours) > 0, large_contours

def detect_net_motion(frame1, frame2, net_rect, net_thresh, net_min_area):
    contours = get_motion_contours(frame1, frame2, net_rect, net_thresh)
    large_contours = [c for c in contours if cv2.contourArea(c) >= net_min_area]
    max_area = max([cv2.contourArea(c) for c in large_contours], default=0)
    return len(large_contours) > 0, max_area, large_contours
