import cv2
import numpy as np
import time

print("Start")
start = time.time()

img1 = cv2.imread('yard-08.png')
img2 = cv2.imread('yard-07.png')
img3 = cv2.imread('yard-06.png')
img4 = cv2.imread('yard-05.png')


cv2.namedWindow('before', cv2.WINDOW_NORMAL)
cv2.imshow('before', img1)
cv2.waitKey(0)
cv2.imshow('before', img2)
cv2.waitKey(0)
cv2.imshow('before', img3)
cv2.waitKey(0)
cv2.imshow('before', img4)
cv2.waitKey(0)


# Finds and matches keypoints in two images using SIFT and BFMatcher
# Takes the two images as input (in order from left to right)
# and returns two numpy arrays with the points in each image that match
def match_keypoints(img_1, img_2, method):
    if method == "SIFT":
        det_desc = cv2.xfeatures2d_SIFT.create()
    elif method == "SURF":
        det_desc = cv2.xfeatures2d_SIFT.create()
    else:
        print("The detector and descriptor method is not correct.")
        exit(-1)

    kp1 = det_desc.detect(img_1)
    desc1 = det_desc.compute(img_1, kp1)

    kp2 = det_desc.detect(img_2)
    desc2 = det_desc.compute(img_2, kp2)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    matches1 = bf.match(desc1[1], desc2[1])
    matches2 = bf.match(desc2[1], desc1[1])
    matches = [m for m in matches1 for x in matches2 if m.distance == x.distance]

    img_pt1 = []
    img_pt2 = []
    for match in matches:
        img_pt1.append(kp1[match.queryIdx].pt)
        img_pt2.append(kp2[match.trainIdx].pt)
    img_pt1 = np.array(img_pt1)
    img_pt2 = np.array(img_pt2)

    return img_pt1, img_pt2


# Finds the homography between two images
def homography(img_1, img_2, method):
    img_pt1, img_pt2 = match_keypoints(img_1, img_2, method)
    h, mask = cv2.findHomography(img_pt2, img_pt1, cv2.RANSAC)

    return h


# Stitches together two images using the homography
def stitch(img_1, img_2, method):
    mask = homography(img_1, img_2, method)

    result = cv2.warpPerspective(img_2, mask, (img_1.shape[1] + 1000, img_1.shape[0] + 1000))
    result[0: img_1.shape[0], 0: img_1.shape[1]] = img_1

    grayscale = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    result = result[y:y + h, x:x + w]

    return result


res1 = stitch(img1, img2, "SURF")
res2 = stitch(img3, img4, "SURF")
res = stitch(res1, res2, "SURF")

final = res[:img1.shape[0], :]


print("Finish time: ", time.time() - start)

cv2.imwrite('result1.png', res1)
cv2.imwrite('result2.png', res2)
cv2.imwrite('result.png', res)
cv2.imwrite('final.png', final)

cv2.namedWindow('after', cv2.WINDOW_NORMAL)
cv2.imshow('after', res1)
cv2.waitKey(0)
cv2.imshow('after', res2)
cv2.waitKey(0)
cv2.imshow('after', res)
cv2.waitKey(0)
cv2.imshow('after', final)
cv2.waitKey(0)
