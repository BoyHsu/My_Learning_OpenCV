import cv2

camera = cv2.VideoCapture(1)

camera.grab()
flag = cv2.CAP_OPENNI_DEPTH_MAP
_, imgDepthMap = camera.retrieve()
print(imgDepthMap.dtype)
windowName = 'depth_map'
cv2.imshow(windowName, imgDepthMap)
while cv2.waitKey(1) == -1:
    camera.grab()
    _, imgDepthMap = camera.retrieve(flag)
    cv2.imshow(windowName, imgDepthMap)

