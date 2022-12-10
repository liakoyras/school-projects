import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

import lib

save_location = './'

np.random.seed(42)

# Create data samples 
m1 = [2, 2]
m2 = [-8, 2]

cov1 = [[2, -0.5], [-0.5, 1]]
cov2 = [[1, 0.5], [0.5, 1]]

params = {'mean' : [m1, m2], 'cov': [cov1, cov2]}

data = lib.create_samples(300, 2, [0.5, 0.5], distribution='normal',
                                              params=params)

print(data)

# Plot the data
fig_data, ax_data = plt.subplots(figsize=(9, 8))

colors = {0 : 'tab:red', 1 : 'tab:blue'}

ax_data.scatter(data['x0'], data['x1'], c=data['class'].map(colors))

plt.savefig(save_location+'scatter.png')

"""
Batch Perceptron
"""
# Prepare data
X = data[['x0', 'x1']] # feature columns
y = np.where(data['class'] == 0, -1, 1) # convert class 0 to negative class

# Classsification
perc_weights, perc_errors, epochs = lib.batch_perceptron(X, y, 0.05, 100000)

print("Calculated weights:", perc_weights, "in", epochs, "iterations.")

# Plot decision boundary
plot_x, plot_y = lib.boundary_points(perc_weights, data[['x0', 'x1']]) 

fig_perc, ax_perc = plt.subplots(figsize=(9, 8))

ax_perc.plot(plot_x, plot_y, 'black') # decision boundary
ax_perc.scatter(data['x0'], data['x1'], c=data['class'].map(colors)) # data

ax_perc.set_ylim(-2, 6)
plt.savefig(save_location+'perceptron.png')

"""
Linear SVM
"""
svm = SVC(kernel='linear')
# Fit SVM
svm.fit(data[['x0', 'x1']], data['class'])

# Calculate weights for decision boundary
coefficients = svm.coef_[0]
bias = svm.intercept_[0]
svm_weights = np.concatenate([[bias], coefficients])

sv = svm.support_vectors_
print("Calculated weights:", svm_weights, "in", svm.n_iter_[0], "iterations.")

# Plot decision boundary
plot_x, plot_y = lib.boundary_points(svm_weights, data[['x0', 'x1']])

fig_svm, ax_svm = plt.subplots(figsize=(9, 8))

ax_svm.plot(plot_x, plot_y, 'black') # decision boundary
ax_svm.scatter(data['x0'], data['x1'], c=data['class'].map(colors)) # data
# highlight support vectors
ax_svm.scatter(sv[:,0], sv[:,1],
               s=100, facecolors='none', edgecolors='tab:orange')

ax_svm.set_ylim(-2, 6)
plt.savefig(save_location+'svm.png')

