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
# knn_df = pd.DataFrame(columns=['image_path', 'class', 'predicted_class', 'knn_neighbors', 'vocabulary_words', 'normalization'])
#
# ks = [2, 3, 5, 7, 9, 11, 17, 25, 35, 45, 55, 65]
# for train in os.listdir('train_dbs'):
#     for k in ks:
#         knn_df = test_k_nearest_neighbors('train_dbs/'+train, 'imagedb_test', test_folders, k, sift, knn_df, True)
#
# knn_df.to_csv('results_knn.csv')

start = time.time()
svms_df = pd.DataFrame(columns=['image_path', 'class', 'predicted_class', 'vocabulary_words', 'normalization', 'kernel', 'epsilon'])

n_words = [50, 100, 150, 200, 250, 300, 400, 500, 700]
normalize = [0, 1]
kernels = ['RBF', 'CHI2', 'SIGMOID', 'INTER']
epsilon = [1.e-08, 1.e-06, 1.e-04, 1.e-02, 0.1]
for w in n_words:
    for n in normalize:
        for k in kernels:
            for e in epsilon:
                svms_df = test_svms('svms/', 'imagedb_test', test_folders, (w, n, k, e), sift, svms_df, verbose=True)
                # svms_df.to_csv('results_svm.csv')

svms_df.to_csv('results_svm.csv')
print('Total SVM testing time: ', round(time.time() - start, 2), ' s')
