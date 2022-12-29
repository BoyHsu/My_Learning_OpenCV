import cv2
import numpy as np


class Pedestrian:
    def __init__(self, id, hsv_frame, track_window):
        self.id = id
        self.track_window = track_window
        self.term_crit = \
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)
        x, y, w, h = track_window
        roi = hsv_frame[y:y + h, x:x + w]
        roi_hist = cv2.calcHist([roi], [0], None, [16], [0, 180])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32
        )
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32
        )
        self.kalman.processNoiseCov = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32
        ) * 0.03
        cx = x+w/2
        cy = y+h/2
        self.kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

    def update(self, frame, hsv_frame):
        back_proj = cv2.calcBackProject([hsv_frame], [0], self.roi_hist, [0, 180], 1)
        ret, self.track_window = cv2.meanShift(back_proj, self.track_window, self.term_crit)
        x, y, w, h = self.track_window
        center = np.array([x+w/2, y+h/2], np.float32)
        prediction = self.kalman.predict()
        estimate = self.kalman.correct(center)
        center_offset = estimate[:, 0][:2] - center
        self.track_window = (x + int(center_offset[0]),
                             y + int(center_offset[1]), w, h)
        x, y, w, h = self.track_window
        cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 4, (255, 0, 0), -1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(frame, 'ID: %d' % self.id, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0),
                    1, cv2.LINE_AA)


def main():
    video_file_path = '../videos/pedestrians.avi'
    cap = cv2.VideoCapture(video_file_path)

    bg_subtractor = cv2.createBackgroundSubtractorKNN()
    history_length = 20
    bg_subtractor.setHistory(history_length)

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))

    pedestrians = []
    num_history_frames_populated = 0

    while True:
        grabbed, frame = cap.read()
        if grabbed is False:
            break

        fg_mask = bg_subtractor.apply(frame)
        if num_history_frames_populated < history_length:
            num_history_frames_populated += 1
            continue

        _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        cv2.erode(thresh, erode_kernel, thresh, iterations=2)
        cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

        contours, hier = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        should_initialize_pedestrians = len(pedestrians) == 0
        id = 0
        for c in contours:
            if cv2.contourArea(c) > 500:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame,  (x, y), (x+w, y+h), (0, 255, 0), 1)
                if should_initialize_pedestrians:
                    pedestrians.append(Pedestrian(id, hsv_frame, (x, y, w, h)))
            id += 1

        for pedestrian in pedestrians:
            pedestrian.update(frame, hsv_frame)

        cv2.imshow('Pedestrian Tracked', frame)
        k = cv2.waitKey(110)
        if k == 27:
            break


if __name__ == "__main__":
    main()
