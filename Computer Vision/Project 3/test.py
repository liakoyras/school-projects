"""Image Classifier Test
This script is a driver program using code from module.py in
order to test the algorithms with a certain set of images.

It runs knn for many different parameters and outputs the
results to the results_knn.csv file.

The train and test images must be on separate folders
depending on the class the image belongs.
"""
from module import *

test_folders = ['fighter_jet/', 'motorbike/', 'school_bus/', 'touring_bike/', 'airplane/', 'car_side/']
sift = cv2.xfeatures2d_SIFT.create()
df = pd.DataFrame(columns=['image_path', 'class', 'predicted_class', 'knn neighbors', 'vocabulary_words', 'normalization'])

for train in os.listdir('train_dbs/'):
    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 21, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200]:
        df = test_k_nearest_neighbors(train, 'imagedb_test', test_folders, k, sift, df, True)

df.to_csv('results_knn.csv')

