import cv2

cap = cv2.VideoCapture("uploaded_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    def mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"X: {x}, Y: {y}")

    cv2.imshow("Frame", frame)
    cv2.setMouseCallback("Frame", mouse_click)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
