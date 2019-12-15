"""Image classifier

This script provides the user with the needed finctions to train
and use SVM and k-nn classifiers in order to classify an image
into a number of set categories.
The classification features are encoded according to the BoVW
(Bag of Visual Words) model.

It can be imported to other projects as a module.

The train images must be on separate folders depending on the
class the image belongs.

This script requires that `opencv-contrib-python version 3.4.2.17`
be installed within the Python environment you are running this
script in.
"""

import numpy as np
import cv2
import os
import time


def extract_sift_features(image_path, sift_object):
    """Extracts features for an image using the SIFT algorithm

    Parameters
    ----------
    image_path : str
        The file location of the image
    sift_object : xfeatures2d_SIFT
        The SIFT object that will be used to extract the features

    Returns
    -------
    descriptors: ndarray
        An ndarray containing the descriptors for the image
        Each descriptor is a 128-dimensional vector
    """
    image = cv2.imread(image_path)

    keypoints = sift_object.detect(image)
    descriptors = sift_object.compute(image, keypoints)
    descriptors = descriptors[1]

    return descriptors


def create_feature_database(directory, folders, sift_object):
    """Creates a database with features from every image from a
    dataset

    Parameters
    ----------
    directory : str
        The path of the directory that contains the class folders
    folders : list
        The list that contains the names of class folders
    sift_object : xfeatures2d_SIFT
        The SIFT object to be used with extract_sift_features()

    Returns
    -------
    train_descriptors_db: ndarray
        An ndarray containing the descriptors for all images inside
        the `directory` folder
        Each descriptor is a 128-dimensional vector
    n_images: int
        The number of images used for the train
    """
    n_images = 0
    train_descriptors_db = np.zeros((0, 128))
    for folder in folders:
        folder_path = os.path.join(directory, folder)

        files = os.listdir(folder_path)
        for file in files:
            n_images += 1
            path = os.path.join(folder_path, file)
            desc = extract_sift_features(path, sift_object)
            train_descriptors_db = np.concatenate((train_descriptors_db, desc), axis=0)

    return train_descriptors_db, n_images


def create_vocabulary(k, train_directory, folders, sift_object, filename, verbose=False):
    """Uses k-means clustering in order to create a vocabulary
       for the BoVW model and saves it to a file inside the
       vocabularies directory.

    Parameters
    ----------
    k: int
        The number of clusters k that k-means will find
        Equals to the number of "visual words" in the vocabulary
    train_directory : str
        The path of the directory that contains the class folders
    folders : list
        The list that contains the names of class folders
    sift_object : xfeatures2d_SIFT
        The SIFT object to be used with extract_sift_features()
    filename: str
        The name of the file where the vocabulary will be saved
        The file extension should be .npy (numpy file)
        If this file exists, the function will inform and exit
    verbose: bool
        Set as True if you want progress messages and a summary
        of the vocabulary to be printed
        (False by default)
    """
    if filename in os.listdir('vocabularies'):
        print("This vocabulary already exists. Please, specify a different filename or use the existing vocabulary.")
        exit(-1)
    else:
        if verbose:
            print("Creating feature database from the images...")
            start = time.time()
        descriptors_db, n_imgs = create_feature_database(train_directory, folders, sift_object)

        if verbose:
            db = time.time()
            print("Finished creating the database: ", round(db-start, 2), " s")

        term_crit = (cv2.TERM_CRITERIA_EPS, int(k/2), 0.1)
        trainer = cv2.BOWKMeansTrainer(k, term_crit, 1, cv2.KMEANS_PP_CENTERS)

        if verbose:
            print("Start clustering...")
            start_cl = time.time()
        vocabulary = trainer.cluster(descriptors_db.astype(np.float32))

        if verbose:
            end_cl = time.time()
            print("Finished clustering: ", round(end_cl-start_cl, 2), " s")
            print()
            print("===Vocabulary summary===")
            print("------------------------")
            print("Number of images used: ", n_imgs)
            print("Database size: ", descriptors_db.shape[0])
            print("Number of words: ", k)
            print("Output file: ", filename)
            print("Total time: ", round(end_cl-start, 2), " s")
            print("========================")
            print()

        np.save('vocabularies/'+filename, vocabulary)


def euclidean_distance(vector1, vector2):
    """Calculates the euclidean distance (L2) between two vectors
    If the second argument is an array of vectors, it will return
    the array of distances of the first argument vector from each
    of the second argument vectors.
    Parameters
    ----------
    vector1: float
        The refference vector

    vector2: float or ndarray
        Either the vector for which the distance from vector1
        will be calculated or an array of vectors for which the
        distance from vector1 to each element will be calculated

    Returns
    -------
    distance: float or ndarray
        The distance between the two input vectors or an array of
        distances between the first input vector and all
    """
    if len(vector2.shape) == 1:
        return np.sqrt(np.sum((vector1-vector2) ** 2))
    else:
        return np.sqrt(np.sum((vector1-vector2) ** 2, axis=1))


def encode_bovw_descriptor(descriptors, vocabulary):
    """Creates a BoVW histogram according to a given vocabulary.

    Parameters
    ----------
    descriptors : ndarray
        The descriptors of an image (calculated with a feature
        extracting algorithm like SIFT)
    vocabulary : ndarray
        The BoVW vocabulary

    Returns
    -------
    bovw_descriptor: ndarray
        An ndarray containing the histograms for each of the
        input descriptors
        Each output descriptor is a vector of size equal to the
        number of Visual Words in the vocabulary
    """
    bovw_descriptor = np.zeros((1, vocabulary.shape[0]))
    for d in range(descriptors.shape[0]):
        distances = euclidean_distance(descriptors[d, :], vocabulary)
        index_min = np.argmin(distances)
        bovw_descriptor[0, index_min] += 1

    return bovw_descriptor


def create_train_features(train_directory, folders, vocabulary_path, sift_object, verbose=False):
    """Uses a BoVW vocabulary to encode each image in a dataset.
    Also, it adds an integer label to each descriptor that
    corresponds to the class it belongs.
    The database created is saved on a .npy file, named
    bovw_encoded_descriptors_<number of words in vocabulary>
    This is extracted from the vocabulary path, so use only
    filenames like the ones exported from create_vocabulary

    Parameters
    ----------
    train_directory : str
        The path of the directory that contains the class folders
    folders : list
        The list that contains the names of class folders
    vocabulary_path : str
        The path to the saved BoVW vocabulary
    sift_object : xfeatures2d_SIFT
        The SIFT object that will be used by extract_sift_features
    verbose: bool
        Set as True if you want progress messages and a summary
        of the database created to be printed
        (False by default)

    Returns
    -------
    bovw_descriptors: ndarray
        An ndarray containing the bovw descriptor for each image,
        enriched with the class label
        Each output descriptor is a vector of size equal to the
        number of Visual Words in the vocabulary and a single
        integer value that shows the class
    """
    head, tail = os.path.split(vocabulary_path)
    if tail not in os.listdir(head):
        print("This vocabulary does not exist. Please, specify a different path.")
        exit(-1)
    else:
        vocabulary = np.load(vocabulary_path)
        if verbose:
            print("Accessing train image folders...")
            start = time.time()

        n_images = 0
        bovw_descs = np.zeros((0, vocabulary.shape[0]+1))
        for folder, class_i in zip(folders, range(len(folders))):
            folder_path = os.path.join(train_directory, folder)

            if verbose:
                print("Accessing images in ", folder_path)
                folder_time = time.time()

            files = os.listdir(folder_path)
            for file in files:
                n_images += 1
                path = os.path.join(folder_path, file)
                desc = extract_sift_features(path, sift_object)
                bovw_desc = encode_bovw_descriptor(desc, vocabulary)
                bovw_desc = np.append(bovw_desc, [class_i])
                # The following 2 lines reshape the ndarray to the proper size after append
                # reduces it to a one dimensional array, so that it can be concatenated
                bovw_desc = bovw_desc[:, np.newaxis]
                bovw_desc = bovw_desc.reshape((bovw_desc.shape[1], bovw_desc.shape[0]))
                bovw_descs = np.concatenate((bovw_descs, bovw_desc), axis=0)

            if verbose:
                print("Encoded images in ", folder_path, " in ", round(time.time()-folder_time), " s")

        n_words = [s for s in vocabulary_path if s.isdigit()]
        n_words = ''.join(n_words)
        filename = 'bovw_encoded_descriptors_'+str(n_words)
        if verbose:
            end_cl = time.time()
            print("===Train Database Summary===")
            print("----------------------------")
            print("Number of images used: ", n_images)
            print("Vocabulary used: ", vocabulary_path)
            print("Output file: ", filename)
            print("Total time: ", round(end_cl - start, 2), " s")
            print("========================")
            print()

        np.save('train_dbs/' + filename, bovw_descs)


train_folders = ['fighter_jet/', 'motorbike/', 'school_bus/', 'touring_bike/', 'airplane/', 'car_side/']
sift = cv2.xfeatures2d_SIFT.create()

# for vocab in os.listdir('vocabularies'):
#     create_train_features('imagedb_train/', train_folders, 'vocabularies/'+vocab, sift, True)

# for i in range(50, 501, 50):
#     create_vocabulary(i, 'imagedb_train/', train_folders, sift, 'vocabulary_'+str(i)+'.npy', True)
#
# v1 = np.array([1, 2, 3])
# v2 = np.array([2, 5, 3])
#
# print(v1.shape)
# print(euclidean_distance(v1, v2))

# vocabular = np.load('vocabularies/vocabulary_50.npy')
#
# descr = extract_sift_features('test.jpg', sift)
# bow = encode_bovw_descriptor(descr, vocabular)

print()
