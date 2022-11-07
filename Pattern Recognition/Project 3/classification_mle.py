import numpy as np
import pandas as pd

import lib

np.random.seed(42)

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


def euclidean_classifier(sample, mean, dims):
    """
    Classify a sample using euclidean distance.

    The sample is classified to the class for which the distance from
    the distribution mean is the least.

    Parameters
    ----------
    sample : list or numpy.ndarray of dimensions (1, dims)
        The sample to be classified
    mean : list or numpy.ndarray of dimensions (c, dims)
        The mean feature values for each class c.
    dims : int
        The number of dimensions (features).

    Returns
    -------
    int
        The class index for which the distance is the least.
    """
    sample = np.array(list(sample))
    mean = np.array(mean)
    class_distances = [lib.euclidean_distance(sample, mean[c], dims) \
                        for c in range(len(mean))]

    return np.argmin(class_distances)

def mahalanobis_classifier(sample, mean, cov, dims):
    """
    Classify a sample using mahalanobis distance.

    The sample is classified to the class for which the distance from
    the distribution is the least.

    Parameters
    ----------
    sample : list or numpy.ndarray of dimensions (1, dims)
        The sample to be classified
    mean : list or numpy.ndarray of dimensions (c, dims)
        The mean feature values for each class c.
    cov : list of list of list or numpy.ndarray of dimensions (c, dims, dims)
        The covariance matrices for each class c.
    dims : int
        The number of dimensions (features).

    Returns
    -------
    int
        The class index for which the distance is the least.
    """
    sample = np.array(list(sample))
    mean = np.array(mean)
    cov = np.array(cov)
    class_distances = [lib.mahalanobis_distance(sample, dims, mean[c], cov[c]) \
                        for c in range(len(mean))]

    return np.argmin(class_distances)

def bayesian_classifier(sample, mean, cov, prob, dims):
    """
    Classify a sample using bayesian discriminants.

    The sample is classified to the class for which the value of
    the discriminant is the biggest.

    Parameters
    ----------
    sample : list or numpy.ndarray of dimensions (1, dims)
        The sample to be classified
    mean : list or numpy.ndarray of dimensions (c, dims)
        The mean feature values for each class c.
    cov : list of list of list or numpy.ndarray of dimensions (c, dims, dims)
        The covariance matrices for each class c.
    prob : list or numpy.ndarray of dimensions (1, c)
        The a priori probability for each class
    dims : int
        The number of dimensions (features).

    Returns
    -------
    int
        The class index for which the discriminant value is the biggest.
    """
    sample = np.array(list(sample))
    mean = np.array(mean)
    cov = np.array(cov)
    class_g = [lib.discriminant(sample, dims, mean[c], cov[c], prob[c]) \
                        for c in range(len(prob))]
    
    return np.argmax(class_g)


"""
Experiment 1
"""
# define values
p = [1/3, 1/3, 1/3]

m1 = [0, 0, 0]
m2 = [1, 2, 2]
m3 = [3, 3, 4]

cov1 = cov2 = cov3 = [[0.8, 0.2, 0.1], [0.2, 0.8, 0.2], [0.1, 0.2, 0.8]]

params = {'mean' : [m1, m2, m3], 'cov': [cov1, cov2, cov3]}

# create samples
train = create_samples(10000, 3, p, params=params)
test = create_samples(1000, 3, p, params=params)

train_size = train.shape[0]
test_size = test.shape[0]
print(50*"-")
print("Experiment 1")
print(50*"-")
print("Test data:")
print(test)

"""
Classify with known parameters
"""
# classify with euclidean, mahalanobis and bayesian classifier
euclidean_errors = mahalanobis_errors = bayesian_errors = 0
for sample in test.iterrows():
    pred_e = euclidean_classifier(sample[1][['x0', 'x1', 'x2']],
                                  params['mean'], 3)
    pred_m = mahalanobis_classifier(sample[1][['x0', 'x1', 'x2']],
                                    params['mean'], params['cov'], 3)
    pred_b = bayesian_classifier(sample[1][['x0', 'x1', 'x2']],
                                 params['mean'], params['cov'], p, 3)
    
    # check and count the errors
    true = sample[1]['class']
    if pred_e != true:
        euclidean_errors += 1
    if pred_m != true:
        mahalanobis_errors += 1
    if pred_b != true:
        bayesian_errors += 1

print("--- Known Parameters ---")
print("Euclidean error:  ", euclidean_errors/test_size)
print("Mahalanobis error:", mahalanobis_errors/test_size)
print("Bayesian error:   ", bayesian_errors/test_size)

"""
Estimate parameters
"""
# parameters with mle
mle_m1 = np.array(np.mean(train[['x0', 'x1', 'x2']][train['class'] == 0], axis=0))
mle_m2 = np.array(np.mean(train[['x0', 'x1', 'x2']][train['class'] == 1], axis=0))
mle_m3 = np.array(np.mean(train[['x0', 'x1', 'x2']][train['class'] == 2], axis=0))

mle_cov1 = np.cov(train[['x0', 'x1', 'x2']][train['class'] == 0], rowvar=False)
mle_cov2 = np.cov(train[['x0', 'x1', 'x2']][train['class'] == 1], rowvar=False)
mle_cov3 = np.cov(train[['x0', 'x1', 'x2']][train['class'] == 2], rowvar=False)

mle_params = {'mean' : [mle_m1, mle_m2, mle_m3], 
              'cov': [mle_cov1, mle_cov2, mle_cov3]}

# classify
mle_euclidean_errors = mle_mahalanobis_errors = mle_bayesian_errors = 0
for sample in test.iterrows():
    mle_pred_e = euclidean_classifier(sample[1][['x0', 'x1', 'x2']],
                                  mle_params['mean'], 3)
    mle_pred_m = mahalanobis_classifier(sample[1][['x0', 'x1', 'x2']],
                                    mle_params['mean'], mle_params['cov'], 3)
    mle_pred_b = bayesian_classifier(sample[1][['x0', 'x1', 'x2']],
                                     mle_params['mean'], mle_params['cov'], p, 3)

    true = sample[1]['class']
    if mle_pred_e != true:
        mle_euclidean_errors += 1
    if mle_pred_m != true:
        mle_mahalanobis_errors += 1
    if mle_pred_b != true:
        mle_bayesian_errors += 1


print("--- Parameters with MLE ---")
print("Euclidean error:  ", mle_euclidean_errors/test_size)
print("Mahalanobis error:", mle_mahalanobis_errors/test_size)
print("Bayesian error:   ", mle_bayesian_errors/test_size)

print(50*"-")
print()

"""
Experiment 2
"""
# define values
p2 = [1/6, 1/6, 2/3]

m12 = [0, 0, 0]
m22 = [1, 2, 2]
m32 = [3, 3, 4]

cov12 = [[0.8, 0.2, 0.1], [0.2, 0.8, 0.2], [0.1, 0.2, 0.8]]
cov22 = [[0.6, 0.2, 0.01], [0.2, 0.8, 0.01], [0.01, 0.01, 0.6]]
cov32 = [[0.6, 0.1, 0.01], [0.1, 0.6, 0.1], [0.1, 0.1, 0.6]]

params2 = {'mean' : [m12, m22, m32], 'cov': [cov12, cov22, cov32]}

# create samples
train2 = create_samples(10000, 3, p2, params=params2)
test2 = create_samples(1000, 3, p2, params=params2)

train2_size = train2.shape[0]
test2_size = test2.shape[0]
print(50*"-")
print("Experiment 2")
print(50*"-")
print("Test data:")
print(test2)

"""
Classify with known parameters
"""
# classify
euclidean_errors2 = mahalanobis_errors2 = bayesian_errors2 = 0
for sample in test2.iterrows():
    pred_e2 = euclidean_classifier(sample[1][['x0', 'x1', 'x2']],
                                  params2['mean'], 3)
    pred_m2 = mahalanobis_classifier(sample[1][['x0', 'x1', 'x2']],
                                    params2['mean'], params2['cov'], 3)
    pred_b2 = bayesian_classifier(sample[1][['x0', 'x1', 'x2']],
                                 params2['mean'], params2['cov'], p2, 3)

    true = sample[1]['class']
    if pred_e2 != true:
        euclidean_errors2 += 1
    if pred_m2 != true:
        mahalanobis_errors2 += 1
    if pred_b2 != true:
        bayesian_errors2 += 1

print("--- Known Parameters ---")
print("Euclidean error:  ", euclidean_errors2/test2_size)
print("Mahalanobis error:", mahalanobis_errors2/test2_size)
print("Bayesian error:   ", bayesian_errors2/test2_size)

"""
Estimate parameters
"""
# parameters with mle
mle_m12 = np.array(np.mean(train2[['x0', 'x1', 'x2']][train2['class'] == 0], axis=0))
mle_m22 = np.array(np.mean(train2[['x0', 'x1', 'x2']][train2['class'] == 1], axis=0))
mle_m32 = np.array(np.mean(train2[['x0', 'x1', 'x2']][train2['class'] == 2], axis=0))

mle_cov12 = np.cov(train2[['x0', 'x1', 'x2']][train2['class'] == 0], rowvar=False)
mle_cov22 = np.cov(train2[['x0', 'x1', 'x2']][train2['class'] == 1], rowvar=False)
mle_cov32 = np.cov(train2[['x0', 'x1', 'x2']][train2['class'] == 2], rowvar=False)

mle_params2 = {'mean' : [mle_m12, mle_m22, mle_m32],
              'cov': [mle_cov12, mle_cov22, mle_cov32]}

# classify
mle_euclidean_errors2 = mle_mahalanobis_errors2 = mle_bayesian_errors2 = 0
for sample in test2.iterrows():
    mle_pred_e2 = euclidean_classifier(sample[1][['x0', 'x1', 'x2']],
                                  mle_params2['mean'], 3)
    mle_pred_m2 = mahalanobis_classifier(sample[1][['x0', 'x1', 'x2']],
                                    mle_params2['mean'], mle_params2['cov'], 3)
    mle_pred_b2 = bayesian_classifier(sample[1][['x0', 'x1', 'x2']],
                                     mle_params2['mean'], mle_params2['cov'], p2, 3)

    true = sample[1]['class']
    if mle_pred_e2 != true:
        mle_euclidean_errors2 += 1
    if mle_pred_m2 != true:
        mle_mahalanobis_errors2 += 1
    if mle_pred_b2 != true:
        mle_bayesian_errors2 += 1


print("--- Parameters with MLE ---")
print("Euclidean error:  ", mle_euclidean_errors2/test2_size)
print("Mahalanobis error:", mle_mahalanobis_errors2/test2_size)
print("Bayesian error:   ", mle_bayesian_errors2/test2_size)

