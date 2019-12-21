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


# Train an SVM for each class, using all of the feature
# databases calculated before and a number of different
# kernels and termination epsilons
# kernels = ['RBF', 'CHI2', 'SIGMOID', 'INTER']
# epsilon = [1.e-08, 1.e-06, 1.e-04, 1.e-02, 0.1]
# for folder, class_i in zip(train_folders, range(len(train_folders))):
#     for features in os.listdir('train_dbs'):
#         n_words = int(''.join([s for s in features if s.isdigit()]))
#         name, extension = features.split('.')
#         normalize = name[-1] == 'n'
#         if normalize:
#             n = '1'
#         else:
#             n = '0'
#         path = os.path.join('train_dbs', features)
#         training_set = np.load(path)
#         for k in kernels:
#             for e in epsilon:
#
#                 train_svm(training_set, class_i, k, e, 'svms/svm_'+str(class_i)+'_'+str(n_words)+'words_'+'norm'+n+'_'+k+'_'+str(e))
#                 print('SVM trained: class=', class_i, 'words=', n_words, 'normalize=', normalize, 'kernel=', k, 'epsilon=', e)
