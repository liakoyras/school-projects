import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def create_samples(N, dims, probabilities, distribution='normal', params=None):
    """
    Create data samples in classes with known probability and distribution.
    
    Parameters
    ----------
    N : int
        The total number of samples to create.
    dims : int
        The dimensionality of the data (the created samples will consist
        of vectors of this number of dimensions.
    probabilities : list of float
        The probability that any given sample belongs to the specific
        class. The length of this list is equal to the number of classes.
    distribution : {'normal'}
        The distribution of the samples. Only normal (gaussian)
        distribution is implemented.
    params : dict, optional
        In the case of normal distribution, the dictionary should contain
        the mean and standard deviations of the distributions of all
        classes in the following format: {'mean':[...], 'cov':[...]}.
        If dims > 1, the lists should contain a list for each class that
        contains the values for each feature.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the samples with feature columns named
        x0, x1 ... and a class column that contains the class number.
        The number of the class is equal to the position of the class
        in the probabilities input list.
    
    Raises
    ------
    ValueError
        If a distribution other than 'normal' is provided as input.
    ValueError
        If the number of dimensions given is not a positive integer.
    ValueError
        If the number of classes (size of the probability array) is not
        equal to the size of the distribution paramerer matrices.
    ValueError
        If no parameters are passed for the gaussian distribution.
    """
    if dims < 1 or str(type(dims)) != "<class 'int'>":
        raise ValueError("The number of dimensions must be a positive integer.")
    else:
        if distribution != 'normal':
            raise ValueError("Distributions other than 'normal' are not yet implemented.")
        elif params is not None:
            if len(params['mean']) != len(params['cov']) or len(params['mean']) != len(probabilities) or len(params['cov']) != len(probabilities):
                raise ValueError("The number of mean vectors and covariance matrices provided must be equal to the number of class probabilities.")
        else:
            raise ValueError("Normal distribution was selected but no parameters passes.")

    num_class_samples = [int(p * N) for p in probabilities] # number of samples for each class
    
    if distribution == 'normal' and params is not None:
        output = pd.DataFrame()
        column_names = ['x'+str(d) for d in range(dims)]
        if dims == 1:
            for c, num_samples in enumerate(num_class_samples): # create samples for each class
                class_samples = np.random.normal(params['mean'][c],params['cov'][c], num_samples)
                # save each sample as a pd.DataFrame row
                class_samples = pd.DataFrame(class_samples, columns=column_names)
                class_samples['class'] = c # set class value
                output = pd.concat([output, class_samples])
        else:
            for c, num_samples in enumerate(num_class_samples):
                class_samples = np.random.multivariate_normal(params['mean'][c],params['cov'][c], num_samples)
                class_samples = pd.DataFrame(class_samples, columns=column_names)
                class_samples['class'] = c
                output = pd.concat([output, class_samples])
                
    
    return output.reset_index(drop=True)


def batch_perceptron(X, y, learning_rate=0.01, epochs=1000, weights=None):
    """
    Perform binary classification with batch perceptron.

    The algorithm stops when it is converged (there are no classification 
    errors) or when the maximum number of iterations is reached.
    
    Parameters
    ----------
    X : pandas.DataFrame
        A DataFrame containing the training data.
    y : numpy.ndarray
        The class labels of each sample. Those must be -1 and 1.
    learning_rate : float, default 0.01
        The algorithms learning rate. A multiplier to reduce the
        contribution of each error to the batch correction and the magnitude
        of the correction itself.
    epochs : int, default 1000
        The max number of iterations that will run before terminating.
    weights : list of float, optional
        Precalculated weights for each feature. It does not incude the bias
        term. Must be the same dimensions as each sample in X.

    Returns
    -------
    weights : list of float
        The coefficients of the linear discriminant (including the bias).
    errors : int
        The number of misclassified samples after the algorithm finishes.
    epoch : int
        The number of iterations the algorithm has ran for.

    Raises
    ------
    ValueError
        If X and y do not have the same number of samples.
    ValueError
        If the dimensions of X and the weights vector do not match.
    ValueError
        If the values in y are not only 1 or -1.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y do not have the same number of samples.")
    if (weights is not None) and (len(weights) != X.shape[1]):
        raise ValueError("X and weights vector dimensions do not match")
    if ~np.isin(y, [-1, 1]).all():
        raise ValueError("Class labels must be 1 or -1. Multiclass classification is not supported.")
    
    if weights is None:
        weights = np.ones(X.shape[1]) # initialize weights
    
    weights = np.concatenate([[1], weights]) # add bias term
    epoch = 0
    errors = 100 # dummy value in order to run the first iteration
    while (epoch < epochs) and (errors > 0): # termination criteria
        # at the beginning of each epoch, assume 0 errors and
        # then count how many actually exist
        errors = 0
        grad = np.zeros(weights.shape[0]) # initialize epoch correction
        for sample in X.iterrows():
            x = np.array(sample[1])
            a = np.concatenate([[1], x]) # augment vector
            label = y[sample[0]]
            prediction = np.sign(a.dot(weights))
            if prediction * label < 0: # if the sample was misclassified
                errors += 1
                grad += -learning_rate * label * a # calculate correction
            
        epoch += 1
        weights += -learning_rate * grad # calculate new weights
    
    return weights, errors, epoch


def boundary_points(weights, data, points=1000):
    """
    Calculates a number of points in order to plot a decision boundary.

    Those points will always be in the range of the data.

    Parameters
    ----------
    weights : list of float
        The coefficients of the decision boundary's equation
    data : pandas.DataFrame
        The data that will be on the same plain as the boundary line.
        This is used in order to calculate the range of points for the line.
    points : int, default 1000
        The number of points to calculate.

    Returns
    -------
    x : numpy.ndarray
        An array with the values of the x axis.
    y : numpy.ndarray
        An array with the calculated values for the y axis.
    """
    slope = -1 * (weights[1]/weights[2])
    intercept = -1 * weights[0]/weights[2]
    
    x = np.linspace(np.amin(data), np.amax(data), points)
    y = slope * x + intercept
    
    return x, y


def cross_validation(model, features, target, folds=5):
    """
    Performs stratified cross validation on a dataset.

    It uses sklearn's StratifiedKFold.

    Parameters
    ----------
    model : fitted sklearn estimator
        The classifier that will be used for the predictions.
    features : pandas.DataFrame
        The DataFrame with the feature columns for the samples.
    target : pandas.Series
        The Series containing the target variable.
    folds : int, default 5
        The number of cross validation folds. Will be used by
        sklearn.StratifiedKFold.

    Returns
    -------
    list of float
        The error (percentage of misclassified samples) for each fold.
    """
    skf = StratifiedKFold(folds)
    
    errors = []
    for train_i, test_i in skf.split(features, target):
        X_train, X_test = features.iloc[train_i], features.iloc[test_i]
        y_train, y_test = target.iloc[train_i], target.iloc[test_i]
        
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        error = 1 - accuracy_score(y_test, predictions)
        errors.append(error)
    
    return errors

def multiclass_svm(train, test, target_col='class', kernel='linear', C=1000, method='ovo'):
    """
    Uses SVM to classify samples in more than two classes.

    Parameters
    ----------
    train : pd.DataFrame
        The DataFrame containing the training data (features and labels).
    test : pd.DataFrame
        The DataFrame containing the test data (only features).
        However, a check is in place that drops a test column if its name
        is the one specified  in target_col.
    target_col : string, default 'class'
        The name of the target columnn in the training dataset.
    kernel : {'linear', 'rbf', 'sigmoid', 'poly'}
        The SVM kernel. Will be used by sklearn.SVC.
    C : float, default 1000
        The box constraint for the SVMs. Will be used by sklearn.SVC.
    method : 'ovo'
        The strategy for multiclass classification. Only 'ovo' (one-vs-one)
        is implemented. This trains a classifier for each pair of classes
        and uses majority voting to find the final prediction. Ties are
        resolved naively using the "first" class that was voted (the one
        with a lower index number).

    Returns
    -------
    predicted_class : numpy.ndarray
        The class labels predicted for each test sample.
    results : pd.DataFrame
        A DataFrame containing the results of each one-vs-one classifier.
        The columns are named using the class index of the pair.
    
    Raises
    ------
    ValueError
        If a method other than 'ovo' is given.
    """
    if target_col in test.columns:
        test = test.drop([target_col], axis=1
        print("Warning: Test DataFrame contained a column named", \
              target_col, "which is the name passed for the train \
              class labels. This was dropped in order to continue \
              the classification. Test DataFrame should contain only \
              the features."
              )

    n_classes = train[target_col].nunique()
    
    if method != 'ovo':
        raise ValueError("Only one-vs-one classification is implemented.")
    
    results = pd.DataFrame()
    for c1 in range(1, n_classes+1): # the loops find all class pairs 
        for c2 in range(c1+1, n_classes+1):
            # get training data only from this pair
            train_2 = train[train[target_col].isin([c1, c2])]
            X_train = train_2.drop([target_col], axis=1)
            y_train = train_2[target_col] 
            if target_col in test.columns:
                test = test.drop([target_col], axis=1)

            model = SVC(kernel=kernel, C=C)
            model.fit(X_train, y_train)

            predictions = model.predict(test)
            column_name = str(c1) + "-" + str(c2)
            results[column_name] = predictions
    
    predicted_class = results.mode(axis=1)
    if predicted_class.shape[1] != 1:
        predicted_class = predicted_class[0] # naive tie breaking
    predicted_class = np.array(predicted_class)

    return predicted_class, results

