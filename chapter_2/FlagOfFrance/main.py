import cv2
import numpy as np

width = 300
height = 200
img = np.zeros((height, width, 3), dtype=np.uint8)

blue = [164, 85, 0]
white = [255, 255, 255]
red = [53, 65, 239]

row1 = int(width * 0.33)+1
img[:, : row1] = blue

row0 = row1
row1 += int(width * 0.33)+1
img[:, row0: row1] = white

img[:, row1:] = red


cv2.imwrite("Flag.png", img)
