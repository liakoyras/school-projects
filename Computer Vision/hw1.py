import cv2
import numpy as np

img_nf = cv2.imread('NF7.png', cv2.IMREAD_GRAYSCALE)
img_n = cv2.imread('N7.png', cv2.IMREAD_GRAYSCALE)

print("The size of each image is:")
print("Original: ", img_nf.shape)
print("Noisy: ", img_n.shape)

print()


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

strel = np.ones((3, 3), np.uint8)  # structuring element for closing operation

_, binary_nf = cv2.threshold(img_nf, 54, 255, cv2.THRESH_BINARY)
binary_nf[:, 300:] = cv2.morphologyEx(src=binary_nf[:, 300:], op=cv2.MORPH_CLOSE, kernel=strel)

_, binary_n = cv2.threshold(filtered, 56, 255, cv2.THRESH_BINARY)
binary_n[:, 300:] = cv2.morphologyEx(src=binary_n[:, 300:], op=cv2.MORPH_CLOSE, kernel=strel)
binary_n[:, 0:200] = cv2.morphologyEx(src=binary_n[:, 0:200], op=cv2.MORPH_CLOSE, kernel=strel)


_, contours_nf, _ = cv2.findContours(binary_nf, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

_, contours_n, _ = cv2.findContours(binary_n, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


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


eligible_contours_nf = find_eligible_contours(binary_nf, contours_nf)

eligible_contours_n = find_eligible_contours(binary_n, contours_n)

print("The number of eligible items for each image is:")
print("Original: ", len(eligible_contours_nf))
print("Noisy:", len(eligible_contours_n))

print()


def compare_contour_areas(contours1, contours2):
    print('item img1 img2')
    print('--------------')
    for n in range(len(contours1)):
        print("  "+str("{:02d}".format(n)), int(cv2.contourArea(contours1[n])), int(cv2.contourArea(contours2[n])))


compare_contour_areas(eligible_contours_nf, eligible_contours_n)

bounding_n = []
bounding_nf = []
for n in range(len(eligible_contours_nf)):
    bounding_nf.append(cv2.boundingRect(eligible_contours_nf[n]))
    bounding_n.append(cv2.boundingRect(eligible_contours_n[n]))


def draw_bb_mean_grayscale(image, bounding_boxes, color_bb, color_val):
    integral = cv2.integral(image)
    mean_vals = []
    for n in range(len(bounding_boxes)):
        (x, y, w, h) = bounding_boxes[n]

        count = (w + 1) * (h + 1)
        val_sum = integral[y, x] - integral[y, x + w] - integral[y + h, x] + integral[y + h, x + w]
        mean = round(val_sum[0] / count, 2)
        mean_vals.append(mean)

        cv2.rectangle(image, (x, y), (x + w, y + h), color_bb, 2)

        image = cv2.putText(image, str(n), (x + 5, y + h - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, color_bb)
        image = cv2.putText(image, str(mean), (x + (w//2) - 20, y + (h//2) + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color_val)

    return mean_vals


mean_gray_nf = cv2.imread('NF7.png', cv2.IMREAD_COLOR)
mean_gray_n = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)

mean_vals_nf = draw_bb_mean_grayscale(mean_gray_nf, bounding_nf, (0, 0, 255), (0, 255, 0))
mean_vals_n = draw_bb_mean_grayscale(mean_gray_n, bounding_n, (255, 0, 0), (0, 255, 0))

# cv2.imwrite('img_rec_nf.png', img_rec_nf)
# cv2.imwrite('img_rec_n.png', img_rec_n)

cv2.namedWindow('final', )
cv2.imshow('final', mean_gray_nf)
cv2.waitKey(0)

cv2.namedWindow('final2', )
cv2.imshow('final2', mean_gray_n)
cv2.waitKey(0)

cv2.destroyAllWindows()
