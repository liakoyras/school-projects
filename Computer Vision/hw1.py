import cv2
import numpy as np

img_nf = cv2.imread('NF7.png', cv2.IMREAD_GRAYSCALE)
img_n = cv2.imread('N7.png', cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('image', )
cv2.imshow('image', img_nf)
cv2.waitKey(0)

# cv2.namedWindow('before', )
# cv2.imshow('before', img)
# cv2.waitKey(0)

def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    window = [
        (i, j)
        for i in range(-indexer, filter_size-indexer)
        for j in range(-indexer, filter_size-indexer)
    ]
    index = len(window) // 2
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] = sorted(
                0 if (
                    min(i+a, j+b) < 0
                    or len(data) <= i+a
                    or len(data[0]) <= j+b
                ) else data[i+a][j+b]
                for a, b in window
            )[index]
    return data

filtered = median_filter(img_n, 3)

# cv2.namedWindow('after', )
# cv2.imshow('after', filtered)
# cv2.waitKey(0)

# filtered5 = median_filter(img_n, 5)
# cv2.namedWindow('after5', )
# cv2.imshow('after5', filtered5)
# cv2.waitKey(0)

strel = np.ones((3,3), np.uint8)
open = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, strel)

cv2.namedWindow('aftero', )
cv2.imshow('aftero', open)
cv2.waitKey(0)

# close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, strel)
#
# cv2.namedWindow('afteroc', )
# cv2.imshow('afteroc', close)
# cv2.waitKey(0)

cv2.destroyAllWindows()
