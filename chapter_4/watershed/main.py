import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../../images/5_of_diamonds.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow('threshold', thresh)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
cv2.imshow('opening', opening)

sure_bg = cv2.dilate(opening, kernel, iterations=3)
cv2.imshow('sure_bg', sure_bg)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
sure_fg = sure_fg.astype(np.uint8)
cv2.imshow('sure_fg', sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg)
cv2.imshow('unknown', unknown)

ret, markers = cv2.connectedComponents(sure_fg)
markers += 1

markers[unknown == 255] = 0

markers = cv2.watershed(img, markers)
img_fg = img.copy()
img_fg[markers != 2] = [0, 0, 0]
img_bg = img.copy()
img_bg[markers != 1] = [0, 0, 0]
img[markers == -1] = [0, 255, 0]
# cv2.imshow('result', img)
# cv2.imshow('fg_last', img_fg)
# cv2.imshow('bg_last', img_bg)
# cv2.waitKey()
# cv2.destroyAllWindows()


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
plt.show()