"""
Basic utilities to read the raw data and convert them to a vectorized form.
"""
import numpy as np
import pandas as pd
import os

from skimage.io import imread

from skimage.transform import resize
from skimage.feature import hog

from sklearn.model_selection import train_test_split


def read_images(data_path, class_ignore=[], target_dims=(32,32), grayscale=False):
    """
    Iterate through image data found in different folders.
    
    Each subfolder represents one class.
    
    The directory structure has to be
        data_path/
        ├─ class 1/
        │  ├─ image001.jpg
        │  ├─ image002.jpg
        │  ├─ ...
        ├─ class 2/
        │  ├─ image001.jpg
        │  ├─ image002.jpg
        │  ├─ ...
        ├─ class 3/
        ...
    
    Note that all of the files are considered to be image files, so anything
    else will result in undetermined behavior.
    
    Parameters
    ----------
    data_path : str
        The path that contains the image folders.
    class_ignore : list-like, optional
        The name of folders to ignore when iterating the base directory.
    target_dims : tuple of int, default (32,32)
        The spatial dimensions of the image (not including channels).
    grayscale : bool, default False
        Whether or not to read the images as grayscale.

    Yields
    ------
    image : np.ndarray of shape (target_dims, n_channels)
        A numerical representation of the image pixels in a numpy array.
    label : int
        A number that shows the class. It corresponds to the order returned by
        os.listdir. Since this depends on the filesystem, it is not necessarily
        deterministic. A call to os.listdir is needed at runtime to confirm the
        matching.
    """
    class_paths = [path for path in os.listdir(data_path) if path not in class_ignore]
    
    for label, class_path in enumerate(class_paths):
        full_path = os.path.join(data_path, class_path)
        image_files = os.listdir(full_path)
        for image_file in image_files:
            image_path = os.path.join(full_path, image_file)
            image = imread(image_path, as_gray=grayscale)

            if image.shape[0] != target_dims[0] or image.shape[1] != target_dims[1]:
                image = resize(image, target_dims)
            
            yield image, label
            
            
def extract_features(images, method='pixels', pandas=True):
    """
    Convert image pixel values to a single 1-D vector.
    
    It can use different vectorization methods.
    
    Parameters
    ----------
    images : generator that returns (image, label)
        The generator that loops through all of the raw images.
    method : {'pixels', 'hog'}
        The method to use to vectorize the images.
        - 'pixels' : Flatten the image (create a vector using all pixel values.
        - 'hog' : Flatten the Histogram of Oriented Gradients descriptors.
    pandas : bool, default True
        Whether or not to return a pd.DataFrame (otherwise it will return
        a numpy array).

    Returns
    -------
    np.ndarray or pd.DataFrame of shape (n_images, n_features+1)
        The resulting vectorized image dataset.
    """
    dataset = []
    for image, label in images:
        if method == "pixels":
            image = np.ravel(image)
        elif method == "hog":
            image = hog(image, transform_sqrt=True, channel_axis=-1)            
        
        image = np.append(image, label)
        dataset.append(image)
        
    if pandas:
        dataset = pd.DataFrame(dataset)
        
    return dataset


def train_test_val_split(dataset, train=0.6, val=0.2):
    """
    Split a dataset into train, validation and test sets.
    
    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to split.
    train : float, default 0.6
        The percentage of the training set.
    val : float, default 0.2
        The percentage of the validation set.
    
    Returns
    -------
    list of pd.DataFrame
        A list containing the three split sets. The initial indexing is kept.
    
    Raises
    ------
    ValueError
        If the provided percentages add up to more than 1.
    """
    test = 1 - train - val
    if test <= 0:
        raise ValueError("Train and Validation percentage must add up to less than 1.")
    
    train, test_val = train_test_split(dataset, train_size=train, random_state=42)
    val = val/(test+val)
    val, test = train_test_split(test_val, train_size=val, random_state=42)
    
    return [train, val, test]
