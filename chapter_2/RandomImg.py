import cv2
import numpy as np
import os

randomByte = os.urandom(120_000)
randomByteArr = bytearray(randomByte)
flatNpArr = np.array(randomByteArr)

grayImage = flatNpArr.reshape(300, 400)
cv2.imwrite("RandomGrey.png", grayImage)

bgrImage = flatNpArr.reshape(100, 400, 3)
cv2.imwrite("RandomColor.png", bgrImage)