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
and `pandas` be installed within the Python environment you are running this
script in.
"""

import numpy as np
import cv2
import pandas as pd
import os
import time


def extract_sift_features(image_path, sift_object):
    """Extracts features for an image using the SIFT algorithm

    Parameters
    ----------
    image_path : str
        The file location of the image
    sift_object : cv2.xfeatures2d_SIFT
        The SIFT object that will be used to extract the features

    Returns
    -------
    descriptors: numpy.ndarray
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
    sift_object : cv2.xfeatures2d_SIFT
        The SIFT object to be used with extract_sift_features()

    Returns
    -------
    train_descriptors_db: numpy.ndarray
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


def create_vocabulary(k, train_directory, folders, sift_object, verbose=False):
    """Uses k-means clustering in order to create a vocabulary
       for the BoVW model and saves it to a file inside the
       vocabularies directory.
       This folder needs to be created manually before running
       the finction for the first time.
       The name of the file is vocabulary_k.npy, where k is
       the number of words that it will contain.


    Parameters
    ----------
    k: int
        The number of clusters k that k-means will find
        Equals to the number of "visual words" in the vocabulary
    train_directory : str
        The path of the directory that contains the class folders
    folders : list
        The list that contains the names of class folders
    sift_object : cv2.xfeatures2d_SIFT
        The SIFT object to be used with extract_sift_features()
    verbose: bool
        Set as True if you want progress messages and a summary
        of the vocabulary to be printed or False to disable them
        (False by default)
    """
    filename = 'vocabulary_' + str(k) + '.npy'
    if filename not in os.listdir('vocabularies'):
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
    else:
        print("This vocabulary already exists. Please, specify a different filename or use the existing vocabulary.")
        exit(-1)


def euclidean_distance(vector1, vector2):
    """Calculates the euclidean distance (L2) between two vectors
    If the second argument is an array of vectors, it will return
    the array of distances of the first argument vector from each
    of the second argument vectors.
    Parameters
    ----------
    vector1: float
        The refference vector

    vector2: numpy.ndarray
        Either the vector for which the distance from vector1
        will be calculated or an array of vectors for which the
        distance from vector1 to each element will be calculated

    Returns
    -------
    distance: float or numpy.ndarray
        The distance between the two input vectors or an array of
        distances between the first input vector and all of the
    """
    if len(vector2.shape) == 1:
        return np.sqrt(np.sum((vector1-vector2) ** 2))
    else:
        return np.sqrt(np.sum((vector1-vector2) ** 2, axis=1))


def encode_bovw_descriptor(descriptors, vocabulary, normalize=True):
    """Creates a BoVW histogram according to a given vocabulary.

    Parameters
    ----------
    descriptors : numpy.ndarray
        The descriptors of an image (calculated with a feature
        extracting algorithm like SIFT)
    vocabulary : numpy.ndarray
        The BoVW vocabulary
    normalize : bool
        Set to False in order to disable L2 norm normalization
        or to True to allow it
        (True by default)
    Returns
    -------
    bovw_descriptor: numpy.ndarray
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

    if normalize:
        norm = np.sqrt(np.sum(bovw_descriptor ** 2))
        bovw_descriptor = bovw_descriptor / norm

    return bovw_descriptor


def create_train_features(train_directory, folders, vocabulary_path, sift_object, normalize=True, verbose=False):
    """Uses a BoVW vocabulary to encode each image in a dataset.
    Also, it adds an integer label to each descriptor that
    corresponds to the class it belongs.
    The database created is saved on a .npy file, named
    bovw_encoded_descriptors_<number of words in vocabulary>
    It will also have a _n in the end of the name if L2 norm
    normalization was used.
    This is extracted from the vocabulary path, so use only
    filenames like the ones exported from create_vocabulary.
    If the filename exists, the function will not do anything.

    Parameters
    ----------
    train_directory : str
        The path of the directory that contains the class folders
    folders : list
        The list that contains the names of class folders
    vocabulary_path : str
        The path to the saved BoVW vocabulary
    sift_object : cv2.xfeatures2d_SIFT
        The SIFT object that will be used by extract_sift_features
    normalize : bool
        It will be used by encode_bovw_descriptor to allow L2 norm
        normalization
        Set to False to disable
        (True by default)
    verbose: bool
        Set as True if you want progress messages and a summary
        of the database created to be printed
        (False by default)
    """
    head, tail = os.path.split(vocabulary_path)
    if tail not in os.listdir(head):
        print("This vocabulary does not exist. Please, specify a different path.")
        exit(-1)
    else:
        n_words = [s for s in vocabulary_path if s.isdigit()]
        n_words = ''.join(n_words)
        filename = 'bovw_encoded_descriptors_' + str(n_words)

        if normalize:
            filename = filename + '_n.npy'
        else:
            filename = filename + '.npy'
        if filename not in os.listdir('train_dbs/'):
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
                    bovw_desc = encode_bovw_descriptor(desc, vocabulary, normalize)
                    bovw_desc = np.append(bovw_desc, [class_i])
                    # The following 2 lines reshape the ndarray to the proper size after append
                    # reduces it to a one dimensional array, so that it can be concatenated
                    bovw_desc = bovw_desc[:, np.newaxis]
                    bovw_desc = bovw_desc.reshape((bovw_desc.shape[1], bovw_desc.shape[0]))
                    bovw_descs = np.concatenate((bovw_descs, bovw_desc), axis=0)

                if verbose:
                    print("Encoded images in ", folder_path, " in ", round(time.time()-folder_time), " s")

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
        else:
            print('This training database already exists. Try using different parameters.')
            exit(-1)


def k_nearest_neighbors(train_set, test_row, num_neighbors):
    """Uses the k Nearest Neighbors algorithm in order to
    classify a row according to a training set.
    The algorithm uses Euclidean Distance as a measure
    and assumes that the last element of each row in the
    training set is the class identifier.

    Parameters
    ----------
    train_set : numpy.ndarray
        The training set containing the encoded descriptors
        The last element of each descriptor is assumed to be
        the class identifier
    test_row : numpy.ndarray
        A one dimensional ndarray that will be checked against
        the training set
    num_neighbors : int
        The number of neighbors for the knn algorithm

    Returns
    -------
    prediction: int
        The predicted class identifier
    """
    distances = []
    for train_row_index in range(0, train_set.shape[0]):
        train_row = train_set[train_row_index]
        distance = euclidean_distance(test_row, train_row[:-1])
        distances.append((train_row, distance))
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])

    neighbor_classes = [row[-1] for row in neighbors]
    prediction = max(set(neighbor_classes), key=neighbor_classes.count)

    return prediction


def test_k_nearest_neighbors(train_set_path, test_directory, test_folders, num_neighbors, sift_object, result_dataframe, verbose=False):
    """Uses the k Nearest Neighbors algorithm to test an
    image test set according to a training set encoded with
    create_train_features.
    This training set must be stored in a .npy file.
    It extracts the parameters the training set was encoded
    with (number of vocabulary words, normalization) from
    the training set filename, in order to do the same
    encoding to the test set, so it will not work with
    training sets encoded with specifications other than
    those of create_train_features.
    The function returns a pandas DataFrame containing the
    classification results.
    In order for these results to be accurate, the test
    folders must be provided with the same order as they
    were provided in the training folders parameter of
    other functions (for example create_train_features).

    Parameters
    ----------
    train_set_path : str
        The path from which the encoded training set will
        be loaded
    test_directory : str
        The path of the directory that contains the test
        class folders
    test_folders : list
        The list that contains the names of class folders
        of the test set
        They must be with the same order
    num_neighbors : int
        The number of nearest neighbors to be used for knn
    sift_object : cv2.xfeatures2d_SIFT
        The SIFT object that will be used by extract_sift_features
    result_dataframe : pandas.core.frame.DataFrame
        The pandas Dataframe that will contain the results of
        the knn classification
        It needs to have the following columns:
        ['image_path', 'class', 'predicted_class', 'knn neighbors',
         'vocabulary_words', 'normalization']
    verbose: bool
        Set as True if you want progress messages and a summary
        of algorithm results
        (False by default)

    Returns
    -------
    result_dataframe : pandas.core.frame.DataFrame
        The modified result DataFrame that contains the results
        for classification with these parameters
    """
    head, tail = os.path.split(train_set_path)
    if tail not in os.listdir(head):
        print("This dataset does not exist. Please, specify a different path.")
        exit(-1)

    train = np.load(train_set_path)
    n_words = int(''.join([s for s in train_set_path if s.isdigit()]))
    voc_path = 'vocabularies/vocabulary_' + str(n_words) + '.npy'
    name, extension = train_set_path.split('.')
    normalize = name[-1] == 'n'
    vocabulary = np.load(voc_path)

    if verbose:
        print('Loaded training set ', train_set_path)
        print("Accessing test image folders...")
        start = time.time()

    n_images = 0
    n_correct = 0
    for folder, class_i in zip(test_folders, range(len(test_folders))):
        folder_path = os.path.join(test_directory, folder)

        files = os.listdir(folder_path)
        for file in files:
            n_images += 1
            path = os.path.join(folder_path, file)
            desc = extract_sift_features(path, sift_object)
            bovw_desc = encode_bovw_descriptor(desc, vocabulary, normalize)
            prediction = k_nearest_neighbors(train, bovw_desc, num_neighbors)
            if prediction == class_i:
                n_correct += 1

            result_dataframe = result_dataframe.append(pd.Series([path, class_i, prediction, num_neighbors, n_words, normalize], index=result_dataframe.columns), ignore_index=True)

    if verbose:
        print('K Nearest Neighbors training completed.')
        print()
        print('===========Summary===========')
        print('vocabulary=: ', voc_path)
        print('n_words=', n_words)
        print('normalize=', normalize)
        print('k= ', num_neighbors)
        print('Size of training set: ', train.shape[0])
        print('Number of test pictures: ', n_images)
        print('Number of pictures correctly classified: ', n_correct)
        print('Test accuracy: ', round(n_correct*100/n_images, 4), '%')
        print('Total time: ', round(time.time() - start, 2), ' s')
        print()
        print()

    return result_dataframe


def train_svm(training_set, train_class, kernel, epsilon, file_path):
    """Trains an SVM to classify an image using the
    provided training set.
    The SVM is saved in a file with the given path.

    Parameters
    ----------
    training_set : numpy.ndarray
        A 2D array containing the training set
        The last column is assumed to be the label
    train_class : int
        The unique class identifier that defines
        the class and distinguises it from the
        other classes
    kernel : str
        The kernel that will be used for the SVM
    epsilon : float
        The error margin that will be used as one
        of the SVM termination criteria
    file_path : str
        The path and name of the file to which the
        SVM will be saved
    """
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)

    if kernel == 'RBF':
        svm.setKernel(cv2.ml.SVM_RBF)
    elif kernel == 'CHI2':
        svm.setKernel(cv2.ml.SVM_CHI2)
    elif kernel == 'INTER':
        svm.setKernel(cv2.ml.SVM_INTER)
    elif kernel == 'SIGMOID':
        svm.setKernel(cv2.ml.SVM_SIGMOID)

    svm.setTermCriteria((cv2.TERM_CRITERIA_EPS, 100, epsilon))

    labels = np.array([int((train_class == i)) for i in training_set[:, -1]])
    svm.trainAuto(training_set[:, :-1].astype(np.float32), cv2.ml.ROW_SAMPLE, labels)

    svm.save(file_path)


def load_svm(svm_path, svm_parameters):
    """Loads a pretrained svm from a given directory

    Parameters
    ----------
    svm_path : str
        The path to the file containing the SVM
    svm_parameters : tuple
       A tuple containing the parameters used to train
       the SVMs in the following order:
       (kernel, epsilon)
       kernel : str
       epsilon : float (scientific notation or 0.1)
       These are used to load the SVM into a cv2.ml.SVM
       object with the same characteristics

    Returns
    -------
    svm : cv2.ml.SVM
        The SVM loaded from the file
    """
    svm = cv2.ml.SVM_create()
    # svm.setType(cv2.ml.SVM_C_SVC)
    #
    # if svm_parameters[0] == 'RBF':
    #     svm.setKernel(cv2.ml.SVM_RBF)
    # elif svm_parameters[0] == 'CHI2':
    #     svm.setKernel(cv2.ml.SVM_CHI2)
    # elif svm_parameters[0] == 'INTER':
    #     svm.setKernel(cv2.ml.SVM_INTER)
    # elif svm_parameters[0] == 'SIGMOID':
    #     svm.setKernel(cv2.ml.SVM_SIGMOID)
    #
    # svm.setTermCriteria((cv2.TERM_CRITERIA_EPS, 100, svm_parameters[1]))

    svm = svm.load(svm_path)

    return svm


def svm_one_vs_all(svms_path, folders, test_image_path, train_parameters, sift_object):
    """Uses a number of pretrained SVMs to classify an
    image into the same number of classes using an one
    versus all scheme.
    It deduces a filename for a given set of parameters
    that has the following format:
    'svm_<class identifier>_<vocabulary words number>words_
    norm<0 or 1>_<kernel type>_<termination epsilon>
    It calculates an array that contains the distances
    from the hyperplane of each SVM, so that the
    prediction is the class with the minimum distance
    and returns the predicted class.

    Parameters
    ----------
    svms_path : str
        The path to the directory containing the SVMs
    folders : list
       The list of folders containing images from
       different classes
    test_image_path : str
       The path to the image that we want to classify
       with the SVMs
    train_parameters : tuple
       A tuple containing the parameters used to train
       the SVMs in the following order:
       (number of vocabulary words, normalization,
       kernel, epsilon)
       number of vocabulary words : int
       normalization : 1 or 0
       kernel : str
       epsilon : float
       These are going to be used to encode the test
       image the same way as the descriptors that
       trained the SVM
    sift_object : cv2.xfeatures2d_SIFT
        The object to be used by extract_sift_features

    Returns
    -------
    predicted_class : int
        The unique class identifier predicted for the
        test image
    """
    vocabulary = np.load('vocabularies/vocabulary_'+str(train_parameters[0])+'.npy')
    predictions = []
    for class_i in range(len(folders)):
        filename = 'svm_'+str(class_i)+'_'+str(train_parameters[0])+'words_'+'norm'+str(train_parameters[1])+'_'+train_parameters[2]+'_'+str(train_parameters[3])

        if filename in os.listdir(svms_path):
            svm = cv2.ml.SVM_create()
            svm = svm.load(svms_path + filename)
            # svm = load_svm(svms_path + filename, (train_parameters[2], train_parameters[3]))

            desc = extract_sift_features(test_image_path, sift_object)
            bovw_desc = encode_bovw_descriptor(desc, vocabulary, train_parameters[1] == 1)
            prediction = svm.predict(bovw_desc.astype(np.float32), flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)

            predictions.append(prediction[1])
        else:
            print('There is not an existing SVM file with these parameters.')
            print('Check if you used the correct file name specifications or create an svm using the train_svm function.')
            exit(-1)

    predicted_class = predictions.index(min(predictions))

    return predicted_class


def test_svms(svms_path, test_directory, test_folders, train_parameters, sift_object, result_dataframe, verbose=False):
    """Uses the one versus all SVM scheme to test an
    image test set against pretrained SVMs (created using
    the train_svm function).
    It uses the train parameters provided, in order to do
    the same encoding to the test set.
    The function returns a pandas DataFrame containing the
    classification results.
    In order for these results to be accurate, the test
    folders must be provided with the same order as they
    were provided in the respective class folders parameter
    of other functions (for example svm_one_vs_all).

    Parameters
    ----------
    svms_path : str
        The path to the directory containing the SVMs
    test_directory : str
        The path of the directory that contains the test
        class folders
    test_folders : list
        The list that contains the names of class folders
        of the test set
        They must be with the same order
    train_parameters : tuple
       A tuple containing the parameters used to train
       the SVMs in the following order:
       (number of vocabulary words, normalization,
       kernel, epsilon)
       See help(svm_one_vs_all) for more info
    sift_object : cv2.xfeatures2d_SIFT
        The SIFT object that will be used by extract_sift_features
    result_dataframe : pandas.core.frame.DataFrame
        The pandas Dataframe that will contain the results of
        the knn classification
        It needs to have the following columns:
        ['image_path', 'class', 'predicted_class', 'knn neighbors',
         'vocabulary_words', 'normalization']
    verbose: bool
        Set as True if you want progress messages and a summary
        of the algorithm results to be printed
        (False by default)

    Returns
    -------
    result_dataframe : pandas.core.frame.DataFrame
        The modified result DataFrame that contains the results
        for classification with these parameters
    """
    if verbose:
        print("Accessing test image folders...")
        start = time.time()

    n_images = 0
    n_correct = 0
    for folder, class_i in zip(test_folders, range(len(test_folders))):
        folder_path = test_directory+'/'+folder

        files = os.listdir(folder_path)
        for file in files:
            n_images += 1
            path = folder_path+'/'+file
            prediction = svm_one_vs_all(svms_path, test_folders, path, train_parameters, sift_object)
            if prediction == class_i:
                n_correct += 1

            result_dataframe = result_dataframe.append(
                pd.Series([path, class_i, prediction, train_parameters[0], train_parameters[1] == 1, train_parameters[2], train_parameters[3]],
                          index=result_dataframe.columns), ignore_index=True)

    if verbose:
        print('One vs All SVM prediction completed.')
        print()
        print('===========Summary===========')
        print('n_words=', train_parameters[0])
        print('normalize=', train_parameters[1] == 1)
        print('kernel= ', train_parameters[2])
        print('epsilon= ', train_parameters[3])
        print('Number of test pictures: ', n_images)
        print('Number of pictures correctly classified: ', n_correct)
        print('Test accuracy: ', round(n_correct*100/n_images, 4), '%')
        print('SVM time: ', round(time.time() - start, 2), ' s')
        print()
        print()

    return result_dataframe
