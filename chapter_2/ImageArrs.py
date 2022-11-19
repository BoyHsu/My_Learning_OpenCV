import cv2
import numpy as np

img = np.zeros((3,3), dtype=np.uint8)
print(img)

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
print(img)

img = np.zeros((5,4), dtype=np.uint8)
print(img.shape)

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
print(img.shape)