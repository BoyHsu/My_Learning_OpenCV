import cv2

img = cv2.imread('MyPic.png')

print(img.shape)
print(img.size)
print(img.dtype)

print(img.item(150, 120, 0))
img[150, 120] = [255, 255, 255]
print(img.item(150, 120, 0))
img.itemset((150, 120, 0), 0)
print(img.item(150, 120, 0))

img1 = img.copy()
img1[:, :, 1] = 0
cv2.imwrite("MoveGreen.png", img1)

img2 = img.copy()
slice = img2[:100, :100]
img2[150:250, 150:250] = slice
cv2.imwrite("Copy.png", img2)

