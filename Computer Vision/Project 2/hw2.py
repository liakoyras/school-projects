import cv2
import numpy as np


# Finds and matches keypoints in two images using BFMatcher
# Manually implemented cross checking
# Input: Two OpenCV images in order from left to right
# Returns: Two numpy arrays with the points in each image that match
# Options: method - SIFT (for SIFT detector), SURF (for SURF detector)
def match_keypoints(img_1, img_2, method):
    det_desc = None
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
# Input: Two OpenCV images in order from left to right
# Returns: A numpy array with the homography mask
# Options: method - SIFT (for SIFT detector), SURF (for SURF detector)
def homography(img_1, img_2, method):
    img_pt1, img_pt2 = match_keypoints(img_1, img_2, method)
    h, mask = cv2.findHomography(img_pt2, img_pt1, cv2.RANSAC)

    return h


# Uses a bounding box in order to crop the empty space of an image
# Input: An OpenCV image
# Returns: The cropped image
def crop(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)

    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    x, y, w, h = cv2.boundingRect(cnt)
    img = img[y:y + h, x:x + w]

    return img


# Stitches together two images
# Input: Two OpenCV images in order from left to right
# Returns: A numpy array with the homography mask
# Options: method - SIFT (for SIFT detector), SURF (for SURF detector)
def stitch(img_1, img_2, method):
    mask = homography(img_1, img_2, method)
    result = cv2.warpPerspective(img_2, mask, (img_1.shape[1] + 1000, img_1.shape[0] + 1000))
    result[0: img_1.shape[0], 0: img_1.shape[1]] = img_1

    final = crop(result)

    return final


# Creates a panorama out of four images
# Input: A list of 4 OpenCV images in order from left to right
# Returns: The panorama image and its cropped form
# Options: method - SIFT (for SIFT detector), SURF (for SURF detector)
def panorama(images, method):
    res1 = stitch(images[0], images[1], method)
    res1 = res1[:images[0].shape[0], :-10]

    res2 = stitch(images[2], images[3], method)
    res2 = res2[:images[2].shape[0], :-10]

    res = stitch(res1, res2, method)
    res_cropped = res[:img1.shape[0], :-10]

    return res, res_cropped


img1 = cv2.imread('yard/yard-08.png')
img2 = cv2.imread('yard/yard-07.png')
img3 = cv2.imread('yard/yard-06.png')
img4 = cv2.imread('yard/yard-05.png')


# "SIFT" "SURF"
inp_images = [img1, img2, img3, img4]
final_SIFT, final_SIFT_cropped = panorama(inp_images, "SIFT")
final_SURF, final_SURF_cropped = panorama(inp_images, "SURF")

cv2.imwrite('final_SIFT.png', final_SIFT)
cv2.imwrite('final_SIFT_cropped.png', final_SIFT_cropped)
cv2.imwrite('final_SURF.png', final_SURF)
cv2.imwrite('final_SURF_cropped.png', final_SURF_cropped)

cv2.namedWindow('after', cv2.WINDOW_NORMAL)
cv2.imshow('after', final_SIFT)
cv2.waitKey(0)
cv2.imshow('after', final_SIFT_cropped)
cv2.waitKey(0)
cv2.imshow('after', final_SURF)
cv2.waitKey(0)
cv2.imshow('after', final_SURF_cropped)
cv2.waitKey(0)


