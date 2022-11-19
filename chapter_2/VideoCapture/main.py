import cv2

videoCapture = cv2.VideoCapture(0)

fps = 30
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("fps:", fps)
print("size:", size)

# videoWriter = cv2.VideoWriter('MyOutputVid.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
videoWriter = cv2.VideoWriter('MyOutputVid.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, size)

numFramesRemaining = 10 * fps - 1
success, frame = videoCapture.read()
while success and numFramesRemaining > 0:
    videoWriter.write(frame)
    success, frame = videoCapture.read()
    numFramesRemaining -= 1
