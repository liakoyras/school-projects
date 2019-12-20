"""Image Classifier Test
This script is a driver program using code from module.py in
order to test the algorithms with a certain set of images.

It runs knn for many different parameters and outputs the
results to the results_knn.csv file.

The train and test images must be on separate folders
depending on the class the image belongs.
"""
from module import *

test_folders = ['fighter_jet', 'motorbike', 'school_bus', 'touring_bike', 'airplane', 'car_side']
sift = cv2.xfeatures2d_SIFT.create()
df = pd.DataFrame(columns=['image_path', 'class', 'predicted_class', 'knn_neighbors', 'vocabulary_words', 'normalization'])

# ks = [2, 3, 5, 7, 9, 11, 17, 25, 35, 45, 55, 65]
# for train in os.listdir('train_dbs'):
#     for k in ks:
#         df = test_k_nearest_neighbors('train_dbs/'+train, 'imagedb_test', test_folders, k, sift, df, True)
#
# df.to_csv('results_knn_1.csv')

# print(time.time()-start)
