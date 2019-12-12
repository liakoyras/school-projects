"""Image classifier

This script allows the user to train and use SVM and k-nn classifiers
in order to classify an image into a number of set categories.
The classification features are encoded according to the BoVW (Bag
of Visual Words) model.

The train images must be on separate folders depending on the class.

This script requires that `opencv-contrib-python version 3.4.2.17` be installed within the Python
environment you are running this script in.
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
    """Creates a database with features from every image from a set

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


def create_vocabulary(k, directory, folders, sift_object, filename, verbose=False):
    """Uses k-means clustering in order to create a vocabulary
       for the BoVW model and saves it to a file

    Parameters
    ----------
    k: int
        The number of clusters k that k-means will find
        Equals to the number of "visual words" in the vocabulary
    directory : str
        The path of the directory that contains the class folders
    folders : list
        The list that contains the names of class folders
    sift_object : xfeatures2d_SIFT
        The SIFT object to be used with extract_sift_features()
    filename: str
        The name of the file where the vocabulary will be saved
        The file extension should be .npy (numpy file)
    verbose: bool
        Set as True if you want progress messages and a summary
        of the vocabulary to be printed
        (False by default)
    """
    if filename in os.listdir('/'):
        print("This vocabulary already exists. Please, specify a different filename or use the existing vocabulary.")
        exit(-1)
    else:
        if verbose:
            print("Creating feature database from the images...")
            start = time.time()
        descriptors_db, n_imgs = create_feature_database(directory, folders, sift_object)

        if verbose:
            db = time.time()
            print("Finished creating the database: ", db-start)

        term_crit = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
        trainer = cv2.BOWKMeansTrainer(k, term_crit, 1, cv2.KMEANS_PP_CENTERS)

        if verbose:
            print("Start clustering...")
            start_cl = time.time()
        vocabulary = trainer.cluster(descriptors_db.astype(np.float32))

        if verbose:
            end_cl = time.time()
            print("Finished clustering: ", end_cl-start_cl)
            print()
            print("===Vocabulary summary===")
            print("------------------------")
            print("Number of images used: ", n_imgs)
            print("Database size: ", descriptors_db.shape[0])
            print("Number of words: ", k)
            print("Output file: ", filename)
            print("Total time: ", end_cl-start)

        np.save(filename, vocabulary)


train_folders = ['fighter_jet/', 'motorbike/', 'school_bus/', 'touring_bike/', 'airplane/', 'car_side/']
sift = cv2.xfeatures2d_SIFT.create()

create_vocabulary(100, 'imagedb_train/', train_folders, sift, 'vocabulary.npy', True)
