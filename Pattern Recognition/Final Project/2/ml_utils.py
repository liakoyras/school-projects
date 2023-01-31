"""
Functions to automate Machine Learning processes.
"""

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif

from sklearn.metrics import accuracy_score

ALG_SEED = 42

def train(model, features, target, show_train_accuracy=False):
    """
    Fit a classifier.
    
    The input model should be a scikit-learn classifier
    supporting the .fit() method.
    
    Parameters
    ----------
    model : sklearn classifier object
        The classifier to use.
    features : pandas.DataFrame
        The DataFrame containing the features that the
        classifier will be fitted on.
    target : pandas.Series
        The Series with the target class variable.
    show_train_accuracy : bool, default False
        If True, it prints the accuracy of the model
        on the training data.
        
    Returns
    -------
    sklearn classifier object
        The fitted classifier model.
    """
    fitted_model = model.fit(features, target)
    
    if show_train_accuracy:
        predictions = fitted_model.predict(features)
        print("Train accuracy:", accuracy_score(target, predictions))
    
    return fitted_model


def test(model, test_features, test_target, scaler=None):
    """
    Evaluate predictions of a model with a test set.
    
    It makes predictions for the test set and returns those
    along with some evaluation metrics by using metrics().
    
    It can accept a scaler as a parameter that will be used
    to scale the testing data.
    
    Parameters
    ----------
    model : sklearn classifier object
        The fitted model to be tested.
    test_features : pandas.DataFrame
        The features of the test set.
    test_target : pandas.Series
        The Series with the true class labels of the test set.
    scaler : sklearn scaler object or None, default None
        The scaler that can be used to standardize the testing
        data.
        
    Returns
    -------
    float
        The classification accuracy on the provided test set.
    """
    if scaler:
        test_features = pd.DataFrame(scaler.transform(test_features), columns=test_features.columns)
        
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_target, predictions)

    return accuracy


def reduce_dimensions(train_features, train_target, method='pca', target_dimensions=10, transform=None):
    """
    Reduces the dimensionality of a dataset using different methods.
    
    Parameters
    ----------
    train_features : pd.DataFrame or np.ndarray
        The features of the train set.
    train_target : pd.Series or np.ndarray of shape (n_samples, 1)
        The target variable of the training set.
    method : {'pca', 'mutual-info', 'chi2', 'anova-f'}
        The dimensionality reduction (or feature selection) method.
    target_dimensions : int, default 10
        The number of dimensions the output feature sets are going to have.
    transform : list of array-like, optional
        If a list-like is passed, its elements will be transformed by the
        dimensionality reduction technique and then returned.
        
    Returns
    -------
    reducer : sklearn object that implements .transform()
        The fitted object that can transform a feature set.
    reduced : None or list-like
        If a list of feature sets was passed with the transform argument,
        they will be transformed and returned.
    """
    if method == 'pca':
        reducer = PCA(n_components=target_dimensions, random_state=ALG_SEED)
    elif method == 'mutual-info':
        reducer = SelectKBest(mutual_info_classif, k=target_dimensions)
    elif method == 'chi2':
        reducer = SelectKBest(chi2, k=target_dimensions)
    elif method == 'anova-f':
        reducer = SelectKBest(f_classif, k=target_dimensions)
    
    reducer.fit(train_features, train_target)
    
    if transform is None:
        reduced = None
    else:
        reduced = []
        for element in transform:
            reduced.append(reducer.transform(element))

    return reducer, reduced


def classifier_threshold(classifier, test_data, threshold=0.5):
    """
    Changes a classifier's threshold.
    
    This function only works if the classifier given implements
    .predict_proba(), which returns for each sample a tuple with the
    probability of the sample belonging to each class.
    
    By definition, this makes sense only for binary classification, and
    the implementation is based on this (will not work correctly for more
    than two classes).
    
    Parameters
    ----------
    classifier : sklearn classifier that implements .predict_proba()
        The (fitted) classifier to change the threshold of. 
    test_data : pd.DataFrame or np.ndarray
        The features to make predictions on.
    train_target : pd.Series or np.ndarray of shape (n_samples, 1)
        The target variable of the training set.
    threshold : float, default 0.5
        The probability threshold that the sample belongs to the positive
        class (class 1).
        
    Returns
    -------
    np.ndarrray of shape (n_samples,)
        The model's predictions based on the given threshold.
    """
    predidctions = (classifier.predict_proba(test_data)[:,1] >= threshold).astype(bool)
    
    return predidctions
