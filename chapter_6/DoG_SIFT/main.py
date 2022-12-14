import cv2

img = cv2.imread('../../images/varese.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

cv2.drawKeypoints(img, keypoints, img, (51, 163, 236), cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

cv2.imshow('sift_keypoints', img)
cv2.waitKey()