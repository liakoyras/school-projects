import numpy as np

np.random.seed(42)

def create_samples(N, probabilities, distribution='normal', params=None):
    """
    Create data samples in classes with known probability and distribution.
    
    Parameters
    ----------
    N : int
        The total number of samples to create.
    probabilities : list of float
        The probability that any given sample belongs to the specific
        class. The length of this list is equal to the number of classes.
    distribution : {'normal'}
        The distribution of the samples. Only normal (gaussian)
        distribution is implemented.
    params : dict, optional
        In the case of normal distribution, the dictionary should contain
        the mean and standard deviations of the distributions of all
        classes in the following format: {'mean':[...], 'std':[...]}
    
    Returns
    -------
    list of numpy.ndarray
        A list of the same length as probabilities containing ndarrays with
        the samples created for each class.
    
    Raises
    ------
    ValueError
        If a distribution other than 'normal' is provided as input.
    """
    class_samples = []
    for p in probabilities:
        class_samples.append(int(p * N)) # number of class samples
    
    samples = []
    if distribution == 'normal' and params is not None:
        for c, num_samples in enumerate(class_samples): # create samples for each class
            samples.append(np.random.normal(params['mean'][c],params['std'][c], num_samples))
    else:
        raise ValueError("Distributions other than 'normal' are not yet implemented.")
    
    return samples

def classify(samples, boundaries, negative=True):
    """
    Classify samples in two classes based on given decision boundaries.
    
    Parameters
    ----------
    samples : list of numpy.ndarray
        A list of samples similar to those created by create_samples().
        It must contain two classes.
    boundaries : list of float
        A list containing the points where the decision boundary
        changes. It must contain two points. 
    negative : bool, default True
        True if the values between the boundaries are to be classified
        in the negative class (ie. the second one provided in the
        samples list).
    
    Returns
    -------
    dict
        {'sample': list of float,
         'true': list of {0, 1},
         'predicted': list of {0, 1}}
        A dictionary containing each sample, the class they belong to
        and the class according to the decision boundaries.

    Raises
    ------
    ValueError
        If the number of points in the boundary list is not 2.
    ValueError
        If the samples list does not contain 2 classes.
    """   
    if len(boundaries) != 2:
        raise ValueError("Only two boundaries can be provided.")

    if len(samples) != 2:
        raise ValueError("The samples can belong only to two classes.")
    
    results = {'sample': [], 'true': [], 'predicted': []}
    for class_number, class_samples in enumerate(samples):
        for sample in class_samples:
            results['sample'].append(sample)
            results['true'].append(class_number) # keep the true class
            if negative:
                if sample < boundaries[0] or sample > boundaries[1]:
                    results['predicted'].append(0)
                else:
                    results['predicted'].append(1)
            else:
                if sample > boundaries[0] and sample < boundaries[1]:
                    results['predicted'].append(0)
                else:
                    results['predicted'].append(1)
    
    return results

def calculate_confusion_matrix(classification_results):
    """
    Calculate the probabilities of the confusion matrix.
    
    The output matrix will contain the probability that a sample
    belonging to a class was classified correctly or not, and not
    the absolute values of the samples in each category.
    
    The first class (class 0) is considered to be the positive one
    and class 1 is the negative.
    
    Parameters
    ----------
    classification_results : dict
        A dict of the same format returned by classify(), containing the
        true and predicted classes of the samples.
    
    Returns
    -------
    list of list of float
        A 2x2 matrix containing the probabilities of correct and false
        classification for the two classes.
    """
    TP, FP, TN, FN = 0, 0, 0, 0
    c0_size , c1_size = 0, 0 # these will hold the number of samples in each class
    for true, predicted in zip(classification_results['true'], classification_results['predicted']):
        if true == 0:   # the correct class is 0
            c0_size += 1
            if predicted == 0: 
                TP += 1 # correct prediction
            elif predicted == 1:
                FN += 1 # false prediction
        elif true == 1: # the correct class is 1
            c1_size += 1
            if predicted == 0:
                FP += 1 # false prediction
            elif predicted == 1:
                TN += 1 # correct prediction

    return [[TP/c0_size, FN/c0_size], [FP/c1_size, TN/c1_size]] 


def calculate_cost(lambdas, result, prior):
    """
    Parameters
    ----------
    lambdas : list of list of (int or float)
        A 2x2 matrix with the coefficients that weigh the importance of
        a correct or wrong prediction
    result : list of list of float
        The probability confusion matrix, as calculated by
        calculate_confusion_matrix().
    prior : list of float
        A list containing the prior probabilities that a sample belongs
        in each class.

    Returns
    -------
    float
        The cost of the classification decisions.
    """

    cost = prior[0] * (lambdas[0][0]*result[0][0] + lambdas[0][1]*result[0][1]) +\
           prior[1] * (lambdas[1][0]*result[1][0] + lambdas[1][1]*result[1][1])

    return cost


"""
Run example experiment
"""
experiment_samples = 1000000

class_probabilities = [1/3, 2/3]
gaussian_parameters = {'mean': [2, 1.5], 'std': [np.sqrt(0.5), np.sqrt(0.2)]}
samples = create_samples(experiment_samples, class_probabilities, distribution='normal',  params=gaussian_parameters)

boundaries = [0.403, 1.9303]
results = classify(samples, boundaries, negative=True)

probability_matrix = calculate_confusion_matrix(results)

print("Probability to classify correctly a class 0 sample:", probability_matrix[0][0])
print("Probability to classify falsely a class 0 sample:  ", probability_matrix[0][1])
print("Probability to classify falsely a class 1 sample:  ", probability_matrix[1][0])
print("Probability to classify correctly a class 1 sample:", probability_matrix[1][1])

l = [[1, 2], [3, 1]]
cost = calculate_cost(l, probability_matrix, class_probabilities)

print("Cost:", cost)

