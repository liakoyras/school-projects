# Computer Vision
The projects I did for my 9th semester Pattern Recognition course.<br>
I used Python and specifically the libraries numpy, pandas and keras.

The code contains explanatory comments that show the thought process for scripts and docstrings that describe the functions that implement an algorithm or another data analysis process.

## Project 1 (Probabilities-Optimization)
The scripts have the following functionalities:
- `one_die.py` and `two_dice.py`: Simulate fair dice rolls with one and two dice respectively, calculating histograms for the distribution of the results  
- `gradient_descent.py`: Implement the gradient descent algorithm to find a local minimum of a function
- `newton_method.py`: Implement the Newton-Raphson method to find a local minimum
 
## Project 2 (Bayesian Classification)
This script can create sample data belonging in two classes with given gaussian distributions and then use the Bayes theorem to classify the samples and calculate the confusion matrix.

## Project 3 (Bayesian Estimators)
Those scripts build upon the code of the previous project, generating more complicated data and then use the maximum likelihood estimation method to compare classifiers based on the euclidean and mahalanobis distance with the ideal Bayesian classifier.

## Project 4 (Linear Discriminants & SVM)
The code in `lib.py` implements helper functions and classification algorithms that are then used for the following scripts:
- `linear_classifiers.py`: Compare the results and the decision boundary found by a batch perceptron algorithm and a linear SVM algorithm for two generated classes
- `svm_classification.py`: Compare linear and non-linear SVM kernels for binary and multi-class classification on the [Iris Seed Dataset](https://archive.ics.uci.edu/ml/machine-learningdatabases/
00236/seeds_dataset.txt) using cross validation

## Project 5 (Clustering)
The script performs Silhouette Coefficient Analysis for the Iris Seed Dataset and K-means clustering, while comparing different distance metrics.

## Project 6 (Neural Networks)
The script uses a simple neural network to classify the [Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris) and attempts to fine tune the hyperparameters of the network and change its architecture to achieve better results.

## Final Project (Image Classification)
For Part 1 of the project, I had to use a Convolutional Neural Network and test various architectures and techniques (fully convolutional, batch normalization, hyperparameter tuning) to increase the classification accuracy on the [CIFAR-10](https://www.cs.toronto.edu/%7Ekriz/cifar.html) dataset without signs of overfitting.

For Part 2 of the project, the task was to classify images based on the use of a protective face mask, using classical machine learning techniques an dimensionality reduction.<br>
After finding the best dimensionality reduction technique and best classification algorithms using the validation set, I fine-tuned classifier hyperparameters and used classifier stacking to improve the classification accuracy.<br>
Finally, the model was tested with edge cases (incorrect use of face mask), and the effect of changing the classification threshold with the tradeoffs it entails is studied and discussed.
  
