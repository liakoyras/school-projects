# Computer Vision
The projects I did for my 7th semester Computer Vision course.<br>
I used Python and more specifically OpenCV, pandas and keras.

More specifically:

## Project 1 (Introduction to OpenCV)
Given an image of cells and the same image with artificially added salt and pepper noise, we need to:
- Create a proper filter in order to remove the noise from the second image (a median filter was chosen)
- Count the number of items that are inside the image boundaries (the whole cell has to be part of the image in order to be counted).
- Count the area (in pixels) of each item
- Calculate the average grayscale value for the pixels inside the bounding box of each item, using a method for which the execution speed remains constant independently of the box size (I did this using integral images)

## Project 2 (Image stitching)
Using OpenCV functions we need to:
- Create a panorama using sets of four images, extracting features for matching using both SIFT and SURF and compare the two algorithms.<br>
- Print the matching points between each image


## Project 3 (Image classification with K-NN and SVMs)
Given a dataset with labeled images of different vehicles (from the Caltech-256 dataset):
- Extract features from each image using SIFT and train a Bag of Visual Words vocabulary with K-Means clustering
- Create a histogram descriptor for each image using the vocabulary trained above (without using OpenCV functions)
- Create a K-Nearest Neighbors classifier
- Train the knn classifier and an SVM one-versus-all classifier
- Test the accuracy for different values of the hyperparameters

The functions are implemented in module.py and the training and testing scripts in train.py and test.py respectively. The analysis of the accuracy is in the results_analysis.ipynb Jupyter notebook.

## Project 4 (Image classification with neural networks)
This project was in the form of a Kaggle competition.<br>
We need to create a convolutional neural network  that classifies  vehicle images into one of 6 classes.
