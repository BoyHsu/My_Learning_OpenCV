import cv2
from matplotlib import pyplot as plt

img0 = cv2.imread('../../images/nasa_logo.png', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('../../images/kennedy_space_center.jpg', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()
kp0, des0 = orb.detectAndCompute(img0, None)
kp1, des1 = orb.detectAndCompute(img1, None)

# 蛮力匹配
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des0, des1)
# img_matches = cv2.drawMatches(
#     img0, kp0, img1, kp1, matches[:25], img1,
#     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
# )
# plt.imshow(img_matches)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
pairs_of_matches = bf.knnMatch(des0, des1, k=2)
pairs_of_matches = sorted(pairs_of_matches, key=lambda x: x[0].distance)

# knn
# img_pairs_of_matches = cv2.drawMatchesKnn(
#     img0, kp0, img1, kp1, pairs_of_matches[:25], img1,
#     flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img_pairs_of_matches)

# ratio test
matches = [x[0] for x in pairs_of_matches
           if len(x) > 1 and x[0].distance < 0.8 * x[1].distance]
img_matches = cv2.drawMatches(img0, kp0, img1, kp1, matches[:25], img1,
                              flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img_matches)

plt.show()
