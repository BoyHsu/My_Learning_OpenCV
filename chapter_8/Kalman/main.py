import cv2
import numpy as np

img = np.zeros((800, 800, 3), np.uint8)

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0]], np.float32
)
kalman.transitionMatrix = np.array(
    [[1, 0, 1, 0],
     [0, 1, 0, 1],
     [0, 0, 1, 0],
     [0, 0, 0, 1]], np.float32
)
kalman.processNoiseCov = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]], np.float32
) * 0.03

last_measurement = None
last_prediction = None


def on_mouse_moved(event, x, y, flags, param):
    global img, kalman, last_measurement, last_prediction

    measurement = np.array([[x], [y]], np.float32)
    if last_measurement is None:
        kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)
        kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
        prediction = measurement
    else:
        kalman.correct(measurement)
        prediction = kalman.predict()

        cv2.line(img, (int(last_measurement[0]), int(last_measurement[1])),
                 (int(measurement[0]), int(measurement[1])), (0, 255, 0))

        cv2.line(img, (int(last_prediction[0]), int(last_prediction[1])),
                 (int(prediction[0]), int(prediction[1])), (0, 0, 255))

    last_prediction = prediction.copy()
    last_measurement = measurement


cv2.namedWindow('kalman_tracker')
cv2.setMouseCallback('kalman_tracker', on_mouse_moved)

while True:
    cv2.imshow('kalman_tracker', img)
    k = cv2.waitKey(1)
    if k == 27:
        cv2.imwrite('kalman.png', img)
        break
