"""Image Classifier Train
This script is a driver program using code from module.py in
order to train the algorithms with a certain set of images.

It creates vocabularies with different word counts.

The train images must be on separate folders depending on the
class the image belongs.
"""
from module import *

train_folders = ['fighter_jet', 'motorbike', 'school_bus', 'touring_bike', 'airplane', 'car_side']
sift = cv2.xfeatures2d_SIFT.create()

# Create vocabularies with different number of words
# for i in range(150, 251, 100):
#     create_vocabulary(i, 'imagedb_train/', train_folders, sift, True)
#
# for i in range(100, 501, 100):
#     create_vocabulary(i, 'imagedb_train/', train_folders, sift, True)
#
# create_vocabulary(700, 'imagedb_train/', train_folders, sift, True)


# Use all these vocabularies to create feature databases
# both with and without using normalization
# for vocab in os.listdir('vocabularies'):
#     create_train_features('imagedb_train/', train_folders, 'vocabularies/' + vocab, sift, True)
#     create_train_features('imagedb_train/', train_folders, 'vocabularies/' + vocab, sift, False)
