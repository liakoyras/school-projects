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


def reduce_dimensions(train_features, train_target, test_features, method='pca', target_dimensions=10):
    """
    Reduces the dimensionality of a dataset using different methods.
    
    Parameters
    ----------
    train_features : pd.DataFrame or np.ndarray
        The features of the train set.
    test_features : pd.DataFrame or np.ndarray
        The features of the test set.
    test_target : pd.Series or np.ndarray
        The 1-D vector with the true class labels of the test set.
    method : {'pca', 'mutual-info', 'chi2', 'anova-f'}
        The dimensionality reduction (or feature selection) method.
    target_dimensions : int, default 10
        The number of dimensions the output feature sets are going to have.
        
    Returns
    -------
    reduced_train : np.ndarray of shape (n_samples, target_dimensions)
        The train set with its dimensions reduced.
    reduced_test : np.ndarray of shape (n_samples, target_dimensions)
        The test set with its dimensions reduced.
    """
    if method == 'pca':
        reducer = PCA(n_components=target_dimensions, random_state=ALG_SEED)
    elif method == 'mutual-info':
        reducer = SelectKBest(mutual_info_classif, k=target_dimensions)
    elif method == 'chi2':
        reducer = SelectKBest(chi2, k=target_dimensions)
    elif method == 'anova-f':
        reducer = SelectKBest(f_classif, k=target_dimensions)
        
    reduced_train = reducer.fit_transform(train_features, train_target)
    reduced_val = reducer.transform(test_features)

    return reduced_train, reduced_val
