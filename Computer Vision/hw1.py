import cv2
import numpy as np

img_nf = cv2.imread('NF7.png', cv2.IMREAD_GRAYSCALE)
img_n = cv2.imread('N7.png', cv2.IMREAD_GRAYSCALE)

print(img_n.shape[0], img_n.shape[1])
cv2.namedWindow('image', )
cv2.imshow('image', img_nf)
cv2.waitKey(0)

# cv2.namedWindow('before', )
# cv2.imshow('before', img)
# cv2.waitKey(0)

def median_filter(data, filter_size):
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

filtered = median_filter(img_n, 5)

# cv2.namedWindow('after', )
# cv2.imshow('after', filtered)
# cv2.waitKey(0)

# filtered5 = median_filter(img_n, 5)
# cv2.namedWindow('after5', )
# cv2.imshow('after5', filtered5)
# cv2.waitKey(0)

strel = np.ones((3,3), np.uint8)
open = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, strel)
close = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, strel)
oc = cv2.morphologyEx(open, cv2.MORPH_CLOSE, strel)

# T, binary_n = cv2.threshold(close, 50, 255, cv2.THRESH_BINARY)
# cv2.imwrite('close.png', binary_n)
# T, binary_n = cv2.threshold(open, 50, 255, cv2.THRESH_BINARY)
# cv2.imwrite('open.png', binary_n)
# T, binary_n = cv2.threshold(oc, 50, 255, cv2.THRESH_BINARY)
# cv2.imwrite('oc.png', binary_n)

# close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, strel)
#
# cv2.namedWindow('afteroc', )
# cv2.imshow('afteroc', close)
# cv2.waitKey(0)

T_nf, binary_nf = cv2.threshold(img_nf, 54, 255, cv2.THRESH_BINARY)
binary_nf[:,300:] = cv2.morphologyEx(binary_nf[:,300:], cv2.MORPH_CLOSE, strel)
T, binary_n = cv2.threshold(open, 56, 255, cv2.THRESH_BINARY)

# cv2.namedWindow('binary_nf', )
# cv2.imshow('binary_nf', binary_nf)
# cv2.waitKey(0)
#
# cv2.namedWindow('binary_n', )
# cv2.imshow('binary_n', binary_n)
# cv2.waitKey(0)

img_cont_nf, contours_nf, hierarchy_nf = cv2.findContours(binary_nf, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_cont_nf, contours_nf, -1, 127, 2)

img_cont_n, contours_n, hierarchy_n = cv2.findContours(binary_n, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_cont_n, contours_n, -1, 127, 2)

cv2.namedWindow('binary_n', )
cv2.imshow('binary_n', binary_n)
cv2.waitKey(0)


cv2.namedWindow('contours_nf', )
cv2.imshow('contours_nf', img_cont_nf)
cv2.waitKey(0)

cv2.namedWindow('contours_n', )
cv2.imshow('contours_n', img_cont_n)
cv2.waitKey(0)

cv2.destroyAllWindows()
