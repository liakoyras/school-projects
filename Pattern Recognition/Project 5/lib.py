import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances, rand_score

from scipy.stats import mode
from scipy.spatial import distance

from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric

def centroid_initialization(n_centroids, data, rng):
    """
    Create random points to use as the initial centers for K-Means.

    Parameters
    ----------
    n_centroids : int
        The number of centroids to create.
    data : pandas.DataFrame
        The data that need to be clustered, in order to specify the range
        that the centroids are allowed to be in.
        This should not contain the target variable.
    rng : numpy.random.Generator
        The random number generator to use.

    Returns
    -------
    numpy.ndarray of shape (n_centroids, n_features)
        An array containing the centroids in each row.
    """
    data_min, data_max = np.min(data, axis=0), np.max(data, axis=0)
    centroids = [rng.uniform(data_min, data_max) 
                 for _ in range(n_centroids)]
    centroids = np.array(centroids)

    return centroids

def kmeans_cluster(data, k, metric='sqeuclidean', init='random', rng=None):
    """
    Clustering using K-Means algorithm with different metrics.

    Since sklearn does not support different metrics, the cosine distance
    is using pyclustering.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to be clustered. This should not contain the target
        variable (if one is present).
    k : int
        The number of clusters to create.
    metric : string, default 'sqeuclidean'
        The metric to use for the distance from the centers.
        The use of non-euclidean metrics is not trivial and there is no
        guarantee of convergence, nevertheless there are implementations.
    init : string or array-like of shape (n_clusters, n_features),
           default 'random'
        The method of center initialization. When 'random' is passed,
        centroid_initalization will be called instead of any native methods
        of the individual libraries that are used for the algorithm, in
        oreder to have consistency.
        When an array-like is passed, this is assumed to contain the
        centroids.
    rng : numpy.random.Generator or int, optional
        If a generator object is passed, it will be used for the centroid
        initialization and clustering.
        If the parameter is an int, it will be used as the random seed for
        a new generator.
        The default behavior creates a new generator without a seed.

    Returns
    -------
    numpy.ndarray of shape (n_samples,)
        The clustering results in the form of an array that contains the
        cluster index that the sample belongs to.

    Raises
    ------
    ValueError
        If a distance other than sqeuclidean or cosine is used.
    """
    # Initialize random number generator
    if 'numpy.random._generator.Generator' in str(type(rng)):
        rng = rng
    elif 'int' in str(type(rng)):
        rng = np.random.default_rng(seed=rng)
    else:
        rng = np.random.default_rng()
    
    # Initialize centroids
    if init == 'random':
        init_centroids = centroid_initialization(k, data, rng)
    else:
        init_centroids = init
    
    # Use different implementations for each metric
    if metric=='sqeuclidean':
        model = KMeans(n_clusters=k, init=init_centroids,
                       n_init=1, max_iter=500)
        model.fit(data)
        predictions = model.predict(data)
    elif metric=='cosine':
        def cosine_distance(x, y): # cosine distance function
            return distance.cosine(x, y) 
        cos_metric = distance_metric(type_metric.USER_DEFINED,
                                     func=cosine_distance) 
        data = data.to_numpy() # pyclustering does not work with df
        
        model = kmeans(data, init_centroids, metric=cos_metric, itermax=500)
        model.process()
        clusters = model.get_clusters() # this will return a list of arrays
                                        # of sample indexes that belong in
                                        # each cluster, so it needs to be
                                        # converted to be consistent with
                                        # sklearn's .predict() method
        
        predictions = np.zeros((data.shape[0],), dtype=int) # init pred arr
        for index, cluster in enumerate(clusters):
            predictions[cluster] = index # cluster indexes to predictions
    else:
        raise ValueError("Only 'sqeuclidean' (squared euclidean) and \
                         'cosine' metrics are implemented")
    
    return predictions


def silhouette_analysis(data, ks, metric='sqeuclidean', rng=None):
    """
    Find the Silhouette coefficient for K-means clustering.

    It calculates a value for the entirety of the clustering (not
    per-sample) and does so for all values of k given in order to be able to
    determine the best number of clusters.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to be analyzed. The DataFrame should contain only the
        features.
    ks : array-like
        The values of k to test.
    metric : string, default 'sqeuclidean'
        The metric to use for both the clustering and the Silhouette
        Coefficient calculations.
    rng : numpy.random.Generator or int, optional
        If a generator object is passed, it will be used for the centroid
        initialization and clustering.
        If the parameter is an int, it will be used as the random seed for
        a new generator.
        The default behavior creates a new generator without a seed.
    
    Returns
    -------
    list of float
        The Silhouette Coefficient for each value of k.
    """
    # Initialize random number generator
    if 'numpy.random._generator.Generator' in str(type(rng)):
        rng = rng
    elif 'int' in str(type(rng)):
        rng = np.random.default_rng(seed=rng)
    else:
        rng = np.random.default_rng()
        
    # Use one set of centers, each time the first k will be used.
    # This is in order to eliminate as much of the variability that the
    # center initialization adds the the analysis as possible.
    init_centroids = centroid_initialization(ks[-1], data, rng)
    coef = []
    for k in ks:
        predictions = kmeans_cluster(data, k, metric=metric,
                                     init=init_centroids[:k], rng=rng)
        
        silhouette = silhouette_score(data, predictions, metric=metric)  
        coef.append(silhouette)
    
    return coef


def match_labels(true_labels, predicted_labels, balanced=True, idxs=None):
    """
    Try to match cluster index labels to the true labeling system.

    The output of clustering algorithms assigns a sample to each cluster,
    with the clusters usually numbered starting from 0. These values do not
    have any specific meaning, so this function will assign those values
    to the labeling system present in , after detecting which class
    corresponds to which cluster label.
    
    It assumes that the clustering was done with at least 50% accuracy,
    since it decides what is the true label based on the most common value
    on the prediction array vs the expected value with the true labeling
    system.

    The use of mode allows for some error margin in the index selection.

    Parameters
    ----------
    true_labels : array-like of shape (n_samples, )
        The correct labels. This array needs to be sorted by class since
        the beginning (before performing the clustering the data need to be
        sorted by label).
        Otherwise, balanced must be set to False and use the idxs param.
    predicted_labels : array-like of shape (n_samples, )
        The predicted cluster indexes.
    balanced : bool, default True
        If True, the classes are assumed to be balanced (exactly equal
        number of samples from each class). If this assumption holds, the
        matching will happen with fewer input needed.
        Otherwise, the idxs parameter will be used.
    idxs : list of list of int, optional
        The list of sample indexes that each class contains.
        Used only when balanced=False.
        Note that this approach is not thoroughly tested.
    
    Returns
    -------
    pred_corrected : numpy.ndarray of (n_samples, )
        The corrected clustering prediction labels.
    cs : list of int
        The label that each cluster index corresponds to.
    """
    n_classes = true_labels.nunique()
    
    cs = []
    ts = []
    if balanced:
        # find number of samples for each class
        c_samples = int(true_labels.shape[0] / n_classes)
        for c_i in range(n_classes):
            c = mode(predicted_labels[c_samples*c_i : \
                                      c_samples*(c_i+1)],
                     keepdims=False).mode
            t = mode(true_labels[c_samples*c_i : c_samples*(c_i+1)],
                     keepdims=False).mode
            cs.append(c) # most common cluster index for this class
            ts.append(t) # most common label for this class
    else:
        for i in idxs:
            c = mode(predicted_labels[i], keepdims=False).mode
            t = mode(true_labels[i], keepdims=False).mode
            cs.append(c)
            ts.append(t)
    
    pred_corrected = np.full((predicted_labels.shape[0],), -1, dtype=int)
    for i, c in enumerate(cs):
        idxs = np.nonzero(predicted_labels == c)
        pred_corrected[idxs] = ts[i] # replace cluster index with label

    return pred_corrected, cs

