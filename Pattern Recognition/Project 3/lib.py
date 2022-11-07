import numpy as np

def discriminant(x, dims, mean, cov, prob):
    """
    Calculate the value of a bayesian discriminant function of a class.

    The function takes as input the distribution parameters and a priori
    probability of the class and the sample vector for which the value
    is calculated..
    
    Parameters
    ----------
    x : numpy.ndarray of dimensions (1, dims)
        The sample to calculate the value for.
    dims : int
        The dimensions of the gaussian distribution (in other words,
        the number of features).
    mean : numpy.ndarray of dimensions (1, dims)
        A vector with the mean values for each feature.
    cov : numpy.ndarray of dimensions (dims, dims)
        The covariance matrix for the distribution.
    prob : float
        The a priory probability of the class i for which the
        discriminant g_i is calculated.

    Returns
    -------
    float
        The value of the disc

    Raises
    ------
    ValueError
        If the number of dimensions given is not a positive integer.
    ValueError
        If the number of dimensions given does not equal to the size of
        the input matrices.
    ValueError
        If the covariance matrix is not square.
    """
    if dims < 1 or str(type(dims)) != "<class 'int'>":
        raise ValueError("The number of dimensions must be a positive integer.")
    if dims > 1 :
        if (dims != len(x)) or (dims != len(mean)) or (dims != len(cov)):
            raise ValueError("The number of dimensions given must correspond to the dimensions of the sample, mean and covariance matrices.")
    
        if len(cov) != len(cov[0]):
            raise ValueError("The covariance matrix is not square.")


    if dims == 1: # special case for 1 dimension
        determinant = abs(cov) # determinant of 1D matrix
        inverse = 1 / cov # inverse of 1D matrix
        g = -0.5 * ((x-mean) * inverse * (x-mean)) - (dims/2) * np.log(2*np.pi) - 0.5 * np.log(determinant) + np.log(prob)
    elif dims > 1:
        determinant = abs(np.linalg.det(cov))
        inverse = np.linalg.inv(cov)
        g = -0.5 * ((x-mean).T).dot(inverse).dot((x-mean)) - (dims/2) * np.log(2*np.pi) - 0.5 * np.log(determinant) + np.log(prob)
    
    return g


def euclidean_distance(x1, x2, dims):
    """
    Calculate the euclidean distance of two points in arbitrary dimensions.
    
    Parameters
    ----------
    x1 : numpy.ndarray of dimensions (1,dims)
        The first point.
    x2 : numpy.ndarray of dimensions (1,dims)
        The second point.
    dims : int
        The number of dimensions of the space the points x1 and x2 exist in.

    Returns
    -------
    float
        The distance between the two points.

    Raises
    ------
    ValueError
        If the number of dimensions given is not a positive integer.
    ValueError
        If the number of dimensions given does not equal to the size of
        the vectors.
    """
    if dims < 1 or str(type(dims)) != "<class 'int'>":
        raise ValueError("The number of dimensions must be a positive integer.")
    if (dims!= len(x1)) or (dims != len(x2)):
        raise ValueError("Input vectors must have the number of dimensions given.")
    
    if dims == 1:
        distance = np.sqrt(np.square(x1[0] - x2[0]))
    elif dims > 1:
        distance = np.sqrt(np.sum(((x1-x2).T).dot(x1-x2)))

    return distance

def mahalanobis_distance(x, dims, mean, cov):
    """
    Calculate the mahalanobis distance between a point and a distribution.

    Parameters
    ----------
    x : numpy.ndarray of dimensions (1, dims)
        The sample to calculate the distance for.
    dims : int
        The dimensions of the gaussian distribution (in other words,
        the number of features).
    mean : numpy.ndarray of dimensions (1, dims)
        A vector with the mean values for each distribution variable.
    cov : numpy.ndarray of dimensions (dims, dims)
        The covariance matrix for the distribution.

    Returns
    -------
    float
        The distance of the point to the distribution.

    Raises
    ------
    ValueError
        If the number of dimensions given is not a positive integer.
    ValueError
        If the number of dimensions given does not equal to the size of
        the input matrices.
    ValueError
        If the covariance matrix is not square.
    """
    if dims < 1 or str(type(dims)) != "<class 'int'>":
        raise ValueError("The number of dimensions must be a positive integer.")
    else:
        if (dims != len(x)) or (dims != len(mean)) or (dims != len(cov)):
            raise ValueError("The number of dimensions given must correspond to the dimensions of the point and the mean and covariance matrices.")

        if len(cov) != len(cov[0]):
            raise ValueError("The covariance matrix is not square.")

    if dims == 1:
        inverse = 1 / cov
        distance = np.sqrt((x-mean) * inverse * (x-mean))
    elif dims > 1:
        inverse = np.linalg.inv(cov)
        distance = np.sqrt(((x-mean).T).dot(inverse).dot((x-mean)))

    return distance

