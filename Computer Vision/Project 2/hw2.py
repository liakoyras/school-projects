import cv2
import numpy as np
import time


# Finds and matches keypoints in two images using BFMatcher
# Manually implemented cross checking
# Input: Two OpenCV images in order from left to right
# Returns: Two numpy arrays with the points in each image that match
# Options: method - SIFT (for SIFT detector), SURF (for SURF detector)
def match_keypoints(img_1, img_2, method):
    if method == "SIFT":
        det_desc = cv2.xfeatures2d_SIFT.create()
    elif method == "SURF":
        det_desc = cv2.xfeatures2d_SURF.create()
    else:
        det_desc = None
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

    f = open("points.txt", "a")
    f.write("Point pairs \n")
    for i1, i2 in zip(img_pt1, img_pt2):
        f.write(str(i1) + "    "+str(i2)+"\n")
    f.write("------------\n")
    f.close()
    return img_pt1, img_pt2


# Finds the homography between two images
# Input: Two OpenCV images in order from left to right
# Returns: A numpy array with the homography mask
# Options: method - SIFT (for SIFT detector), SURF (for SURF detector)
def homography(img_1, img_2, method):
    img_1pt, img_2pt = match_keypoints(img_1, img_2, method)
    h, mask = cv2.findHomography(img_2pt, img_1pt, cv2.RANSAC)

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
#          direction - r2l (for stitching the right image to the left),
#                      l2r (for stitching the left image to the right)
def stitch(left, right, method, direction):
    translation = int(max(left.shape) * 1.5)
    if direction == 'l2r':
        h = homography(right, left, method)
        translation_matrix = np.array([[1, 0, translation], [0, 1, 0], [0, 0, 1]])
        mat = np.matmul(translation_matrix, h)
        result = cv2.warpPerspective(left, mat, (translation*2, translation*2))
        result[0: right.shape[0], translation:translation + right.shape[1]] = right
    elif direction == 'r2l':
        h = homography(left, right, method)
        result = cv2.warpPerspective(right, h, (translation*2, translation*2))
        result[0: left.shape[0], 0: left.shape[1]] = left
    else:
        result = None
        print("The stitching direction was not specified correctly.")
        exit(-2)

    final = crop(result)

    return final


# Creates a panorama out of four images
# Input: A list of 4 OpenCV images in order from left to right
# Returns: The panorama image and its cropped form
# Options: method - SIFT (for SIFT detector), SURF (for SURF detector)
#          direction - r2l (gives the sensation of rotating the camera right to left),
#                      l2r (gives the sensation of rotating the camera left to right)
def panorama(images, method, direction):
    f = open("points.txt", "a")
    f.write("Method: "+method+ "\n")
    f.close()
    print("First stitching")
    f = open("points.txt", "a")
    f.write("First stitching \n")
    f.close()
    res1 = stitch(images[0], images[1], method, 'l2r')
    print("Second stitching")
    f = open("points.txt", "a")
    f.write("Second stitching \n")
    f.close()
    res2 = stitch(images[2], images[3], method, 'r2l')
    print("Third stitching")
    f = open("points.txt", "a")
    f.write("Third stitching \n")
    f.close()
    if direction == 'l2r':
        res2 = res2[:images[2].shape[0], 10:]
        res = stitch(res1, res2, method, direction)
        res_cropped = res[:images[1].shape[0], 20:]
    elif direction == 'r2l':
        res1 = res1[:images[1].shape[0], :-10]
        res = stitch(res1, res2, method, direction)
        res = res[:, :-10]
        res_cropped = res[:images[2].shape[0], :-20]
    else:
        res, res_cropped = None, None
        print("The panorama viewing direction was not specified correctly.")
        exit(-3)

    return res, res_cropped


# Resizes an image
# Input: img - the OpenCV image to be resized
#        scale_percent - the percentage to which the source image will be scaled
# Returns: The resized image
def change_size(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized


img1 = cv2.imread('yard/yard-08.png')
img2 = cv2.imread('yard/yard-07.png')
img3 = cv2.imread('yard/yard-06.png')
img4 = cv2.imread('yard/yard-05.png')

inp_images = [img1, img2, img3, img4]
print("Panorama with yard images")
print("Start SIFT")
start = time.time()
final_SIFT, final_SIFT_cropped = panorama(inp_images, "SIFT", 'l2r')
sift = time.time()
print("SIFT time: ", sift-start)
print("Start SIFT")
final_SURF, final_SURF_cropped = panorama(inp_images, "SURF", 'l2r')
print("SURF time", time.time()-sift)

cv2.imwrite("SIFT.png", final_SIFT)
cv2.imwrite("SURF.png", final_SURF)
cv2.namedWindow('yard', cv2.WINDOW_NORMAL)
cv2.imshow('yard', final_SIFT)
cv2.waitKey(0)

# img1_mine = cv2.imread('mine/1.jpg')
# img2_mine = cv2.imread('mine/2.jpg')
# img3_mine = cv2.imread('mine/3.jpg')
# img4_mine = cv2.imread('mine/4.jpg')
#
# inp_images_mine = [change_size(i, 33) for i in [img1_mine, img2_mine, img3_mine, img4_mine]]
# print("Panorama with my images")
# print("Start SIFT")
# start = time.time()
# final_SIFT_mine, final_SIFT_cropped_mine = panorama(inp_images_mine, "SIFT", 'l2r')
# sift = time.time()
# print("SIFT time: ", sift-start)
# print("Start SIFT")
# final_SURF_mine, final_SURF_cropped_mine = panorama(inp_images_mine, "SURF", 'l2r')
# print("SURF time", time.time()-sift)
#
# cv2.imwrite("SIFT_mine.png", final_SIFT_mine)
# cv2.imwrite("SURF_mine.png", final_SURF_mine)
# cv2.namedWindow('mine', cv2.WINDOW_NORMAL)
# cv2.imshow('mine', final_SIFT_mine)
# cv2.waitKey(0)
