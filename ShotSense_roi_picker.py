import cv2

# load your video
cap = cv2.VideoCapture("uploaded_video.mp4")  # change name if needed

# read first frame
ret, frame = cap.read()

if not ret:
    print("Error: could not read video")
    exit()

# select hoop ROI
hoop_roi = cv2.selectROI("Select Hoop", frame, False)
cv2.destroyWindow("Select Hoop")

# select net ROI
net_roi = cv2.selectROI("Select Net", frame, False)
cv2.destroyWindow("Select Net")

# print values so you can reuse them
print("HOOP ROI:", hoop_roi)
print("NET ROI:", net_roi)

cap.release()
cv2.destroyAllWindows()
