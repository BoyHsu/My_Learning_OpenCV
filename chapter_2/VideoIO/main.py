import cv2
import numpy

videoCapture = cv2.VideoCapture("MyInputVid.avi")

fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("fps:", fps)
print("size:", size)

# videoWriter = cv2.VideoWriter('MyOutputVid.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
videoWriter = cv2.VideoWriter('MyOutputVid.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, size)

success, frame = videoCapture.read()
while success:
    videoWriter.write(frame)
    success, frame = videoCapture.read()
