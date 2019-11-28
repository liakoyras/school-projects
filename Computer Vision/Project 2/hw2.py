import cv2

import numpy as np
import time


print("Start")
start = time.time()

img1 = cv2.imread('yard-08.png')
img2 = cv2.imread('yard-07.png')
img3 = cv2.imread('yard-06.png')
img4 = cv2.imread('yard-05.png')

print(img1.shape)

read_t = time.time()
print("Time to read the files: ", read_t - start)

cv2.namedWindow('before1', cv2.WINDOW_NORMAL)
cv2.imshow('before1', img1)
cv2.waitKey(0)
cv2.namedWindow('before2', cv2.WINDOW_NORMAL)
cv2.imshow('before2', img2)
cv2.waitKey(0)
# cv2.imshow('before', img3)
# cv2.waitKey(0)
# cv2.imshow('before', img4)
# cv2.waitKey(0)

sift_s = time.time()
sift = cv2.xfeatures2d_SIFT.create()

kp1 = sift.detect(img1)
desc1 = sift.compute(img1, kp1)

kp2 = sift.detect(img2)
desc2 = sift.compute(img2, kp2)

sift_e = time.time()
print("SIFT time: ", sift_e-sift_s)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
bf2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches1 = bf.match(desc1[1], desc2[1])
matches2 = bf.match(desc2[1], desc1[1])
matches = [m for m in matches1 for x in matches2 if m.distance == x.distance]
matchescf = bf2.match(desc1[1], desc2[1])

leng = len(matches)
lengcf = len(matchescf)
leng1 = len(matches1)
leng2 = len(matches2)
match_t = time.time()
print("Matching time: ", match_t - sift_e)

img_pt1 = []
img_pt2 = []
for m in matches:
    img_pt1.append(kp1[m.queryIdx].pt)
    img_pt2.append(kp2[m.trainIdx].pt)
img_pt1 = np.array(img_pt1)
img_pt2 = np.array(img_pt2)

M, mask = cv2.findHomography(img_pt2, img_pt1, cv2.RANSAC)
result = cv2.warpPerspective(img2, M, (img1.shape[1]+1000, img1.shape[0]+1000))
result[0: img1.shape[0], 0: img1.shape[1]] = img1

hom_t = time.time()
print("Homography time: ", hom_t - match_t)

cv2.namedWindow('after', cv2.WINDOW_NORMAL)
cv2.imshow('after', result)
cv2.waitKey(0)
