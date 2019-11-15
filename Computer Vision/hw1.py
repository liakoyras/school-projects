import cv2
import numpy as np

img_nf = cv2.imread('NF7.png', cv2.IMREAD_GRAYSCALE)
img_n = cv2.imread('N7.png', cv2.IMREAD_GRAYSCALE)

print(img_nf.shape)
print(img_n.shape)


def median_filter(image, filter_size):
    data = np.copy(image)
    indexer = filter_size // 2
    window = [
        (i, j)
        for i in range(-indexer, filter_size - indexer)
        for j in range(-indexer, filter_size - indexer)
    ]
    index = len(window) // 2
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] = sorted(
                0 if (
                        min(i + a, j + b) < 0
                        or len(data) <= i + a
                        or len(data[0]) <= j + b
                ) else data[i + a][j + b]
                for a, b in window
            )[index]
    return data


filtered = median_filter(img_n, 5)

# cv2.namedWindow('before', )
# cv2.imshow('before', img_n)
# cv2.waitKey(0)
#
# cv2.namedWindow('after', )
# cv2.imshow('after', filtered)
# cv2.waitKey(0)

# cv2.imwrite('filtered.png', filtered)

# filtered5 = median_filter(img_n, 5)
# cv2.namedWindow('after5', )
# cv2.imshow('after5', filtered5)
# cv2.waitKey(0)


# strel = np.ones((3,3), np.uint8)
# open = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, strel)

# cv2.imwrite('opened.png', open)

# close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, strel)
#
# cv2.namedWindow('afteroc', )
# cv2.imshow('afteroc', close)
# cv2.waitKey(0)

_, binary_nf = cv2.threshold(img_nf, 54, 255, cv2.THRESH_BINARY)
strel = np.ones((3, 3), np.uint8)
binary_nf[:, 300:] = cv2.morphologyEx(src=binary_nf[:, 300:], op=cv2.MORPH_CLOSE, kernel=strel)
_, binary_n = cv2.threshold(filtered, 56, 255, cv2.THRESH_BINARY)
binary_n[:, 300:] = cv2.morphologyEx(src=binary_n[:, 300:], op=cv2.MORPH_CLOSE, kernel=strel)
binary_n[:, 0:200] = cv2.morphologyEx(src=binary_n[:, 0:200], op=cv2.MORPH_CLOSE, kernel=strel)

# cv2.imwrite('binary_n.png', binary_n)
# cv2.imwrite('binary_nf.png', binary_nf)
# cv2.imwrite('binary_nf+close.png', binary_nf)

# cv2.namedWindow('binary_nf', )
# cv2.imshow('binary_nf', binary_nf)
# cv2.waitKey(0)
#
# cv2.namedWindow('binary_n', )
# cv2.imshow('binary_n', binary_n)
# cv2.waitKey(0)
_, contours_nf, _ = cv2.findContours(binary_nf, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# contours_nf_img = np.copy(binary_nf)
# cv2.drawContours(contours_nf_img, contours_nf, -1, 127, 2)

_, contours_n, _ = cv2.findContours(binary_n, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# contours_n_img = np.copy(binary_n)
# cv2.drawContours(contours_n_img, contours_n, -1, 127, 2)

# cv2.imwrite('contours_n.png', contours_n_img)
# cv2.imwrite('contours_nf.png', contours_nf_img)

# cv2.namedWindow('binary_nf', )
# cv2.imshow('binary_nf', binary_nf)
# cv2.waitKey(0)

# cv2.namedWindow('contours_nf_img', )
# cv2.imshow('contours_nf_img', contours_nf_img)
# cv2.waitKey(0)
#
# cv2.namedWindow('contours_n_img', )
# cv2.imshow('contours_n_img', contours_n_img)
# cv2.waitKey(0)

def find_eligible_contours(image, contours):
    index_ineligible_contours = []
    width = image.shape[1]
    height = image.shape[0]

    for c in range(len(contours)):
        for i in range(len(contours[c])):
            if 0 in contours[c][i][0] or contours[c][i][0, 0] >= width - 1 or contours[c][i][0, 1] >= height - 1:
                index_ineligible_contours.append(c)
                break

    eligible_contours = [contours[j] for j in range(len(contours)) if j not in index_ineligible_contours]

    return eligible_contours


eligible_contours_n = find_eligible_contours(binary_n, contours_n)

# eligible_contours_n_img = np.copy(binary_n)
# cv2.drawContours(eligible_contours_n_img, eligible_contours_n, -1, 127, 2)


eligible_contours_nf = find_eligible_contours(binary_nf, contours_nf)

# eligible_contours_nf_img = np.copy(binary_nf)
# cv2.drawContours(eligible_contours_nf_img, eligible_contours_nf, -1, 127, 2)

print(len(eligible_contours_n))
print(len(eligible_contours_nf))


def compare_contour_areas(contours1, contours2):
    print('img1 img2')
    print('---------')
    for n in range(len(contours1)):
        print(int(cv2.contourArea(contours1[n])), int(cv2.contourArea(contours2[n])))

compare_contour_areas(eligible_contours_nf, eligible_contours_n)

# cv2.imwrite('eligible_contours_nf_img.png', eligible_contours_nf_img)
# cv2.imwrite('eligible_contours_n_img.png', eligible_contours_n_img)
#
# cv2.namedWindow('eligible_contours_n_img', )
# cv2.imshow('eligible_contours_n_img', eligible_contours_n_img)
# cv2.waitKey(0)

cv2.destroyAllWindows()
