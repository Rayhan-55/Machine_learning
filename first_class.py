import cv2 as cv
import numpy as np
import os
from datetime import datetime

print("This motion detection system")

# Folder creation
if not os.path.exists("Motion capture"):
    os.makedirs("Motion capture")
    print("Motion capture folder created")

# Open camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 500)

if not cap.isOpened():
    print("Camera not found")
else:
    print("Camera found successfully")

for i in range(10):
    cap.read()

ret, frame1 = cap.read()
ret, frame2 = cap.read()

if not ret or frame1 is None or frame2 is None:
    print("Frame read error")
    cap.release()
    exit()
print("Frame read successfully")
print("Motion detection started")

motion_count = 0
save_count = 0
motion_frame = []
max_storage_frame = 5
auto_save = True
save_interval = 30

while True:
    try:
        if frame1 is None or frame2 is None:
            print("Invalid Frame")
            ret, frame1 = cap.read()
            ret, frame2 = cap.read()
            continue

        dif = cv.absdiff(frame1, frame2)
        grey = cv.cvtColor(dif, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(grey, (5, 5), 0)
        _, threst = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
        dilated = cv.dilate(threst, None, iterations=3)
        contour, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        motion_detection = False
        display_frame = frame1.copy()

        for c in contour:
            area = cv.contourArea(c)
            if area > 1000:
                motion_detection = True
                x, y, w, h = cv.boundingRect(c)
                cv.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(display_frame, f"area:{area}", (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if motion_detection:
            motion_count += 1
            cv.putText(display_frame, "Motion detected", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.putText(display_frame, f"Count: {motion_count}", (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if len(motion_frame) >= max_storage_frame:
                motion_frame.pop(0)
            motion_frame.append(display_frame.copy())

            if motion_count % 10 == 0:
                print(f"Motion counted {motion_count}, {datetime.now().strftime('%H:%M:%S')}")

            if auto_save and (motion_count % save_interval == 0):
                time_stamp = datetime.now().strftime("%H-%M-%S")
                filename = f"Motion capture/motion_{time_stamp}.jpg"
                cv.imwrite(filename, display_frame)
                save_count += 1
                print(f"Auto-saved: {filename}")

        # Display and quit key
        h, w = display_frame.shape[:2]
        cv.putText(display_frame, f"Detection: {motion_count}", (10, h - 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv.putText(display_frame, f"Saved: {save_count}", (200, h - 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv.imshow("Motion_detection-main_view", display_frame)
        cv.imshow("Motion_detection-processview", dilated)

        key = cv.waitKey(10) & 0xFF
        if key == ord('q'):
            break

        frame1 = frame2
        ret, frame2 = cap.read()

    except KeyboardInterrupt:
        print("Keyboard interrupt")
        break

cap.release()
cv.destroyAllWindows()
print("Window closed")
