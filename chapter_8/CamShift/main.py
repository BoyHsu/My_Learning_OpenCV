import cv2
import numpy as np

cap = cv2.VideoCapture(0)

for i in range(10):
    success, frame = cap.read()

if not success:
    exit(1)

frame_h, frame_w = frame.shape[:2]

w = frame_w//6
h = frame_h//6
x = frame_w//2 - w//2
y = frame_h//2 - h//2

track_window = (x, y, w, h)

roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

mask = None
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = \
    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)
success, frame = cap.read()
while success:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    rotated_rect, track_window = cv2.CamShift(back_proj, track_window, term_crit)
    box_points = cv2.boxPoints(rotated_rect)
    box_points = np.int0(box_points)
    cv2.polylines(frame, [box_points], True, (255, 0, 0), 2)

    cv2.imshow('back-project', back_proj)
    cv2.imshow('camshift', frame)
    if cv2.waitKey(1) == 27:
        break
    success, frame = cap.read()




