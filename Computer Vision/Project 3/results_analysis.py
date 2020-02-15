# This file is a duplicate of results_analysis.py in order to
# allow for proper language detection. Please see the original file.
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pylab import rcParams

plt.rc('figure', figsize=(15, 8))
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)

def plot_bar_x(labels, data, parameter, title):
    index = np.arange(len(labels))
    plt.bar(index, data)
    plt.xlabel(parameter, fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xticks(index, labels, fontsize=10)
    plt.title(title, fontsize=20)
    plt.show()

knn_df = pd.read_csv('results_knn.csv', index_col='Unnamed: 0')
knn_df.head()

for i in range(knn_df.shape[0]):
    if  knn_df.loc[i, 'predicted_class'] == knn_df.loc[i, 'class']:
        knn_df.loc[i, 'prediction_correct'] = 1
    else:
        knn_df.loc[i, 'prediction_correct'] = 0

knn_df['num_images'] = 1

knn_grouped = knn_df.groupby(['class', 'vocabulary_words', 'normalization', 'knn_neighbors']).sum().drop(columns=['predicted_class'])
knn_grouped


knn_classes = knn_grouped.groupby('class').sum()

classes_folders = ['fighter_jet', 'motorbike', 'school_bus', 'touring_bike', 'airplane', 'car_side']

ind = knn_classes.index.tolist()
for i in range(len(ind)):
    ind[i] = classes_folders[i]

knn_classes.index = ind

knn_classes.loc['average', 'prediction_correct'] = knn_classes['prediction_correct'].sum()
knn_classes.loc['average', 'num_images'] = knn_classes['num_images'].sum()
knn_classes['accuracy'] = knn_classes['prediction_correct'] / knn_classes['num_images']

knn_classes

plot_bar_x(knn_classes.index, knn_classes.accuracy.values, 'Class', 'Accuracy for each class')


knn_words = knn_grouped.groupby('vocabulary_words').sum()

knn_words['accuracy'] = knn_words['prediction_correct'] / knn_words['num_images']

knn_words

plot_bar_x(knn_words.index, knn_words.accuracy.values, 'Number of vocabulary words', 'Accuracy for different number of words in BoVW vocabulary')


knn_norms = knn_grouped.groupby('normalization').sum()

knn_norms['accuracy'] = knn_norms['prediction_correct'] / knn_norms['num_images']

knn_norms

plot_bar_x(knn_norms.index, knn_norms.accuracy.values, 'Normalization', 'Accuracy with and without normalization')


ks = knn_grouped.groupby('knn_neighbors').sum()

ks.loc['average', 'prediction_correct'] = ks['prediction_correct'].sum()
ks.loc['average', 'num_images'] = ks['num_images'].sum()
ks['accuracy'] = ks['prediction_correct'] / ks['num_images']

ks

plot_bar_x(ks.index, ks.accuracy.values, 'k', 'Accuracy for different values of k')


svm_df = pd.read_csv('results_svm.csv', index_col='Unnamed: 0')
svm_df.head()

for i in range(svm_df.shape[0]):
    if  svm_df.loc[i, 'predicted_class'] == svm_df.loc[i, 'class']:
        svm_df.loc[i, 'prediction_correct'] = 1
    else:
        svm_df.loc[i, 'prediction_correct'] = 0

svm_df['num_images'] = 1

svm_grouped = svm_df.groupby(['class', 'vocabulary_words', 'normalization', 'kernel', 'epsilon']).sum().drop(columns=['predicted_class'])
svm_grouped


svm_classes = svm_grouped.groupby('class').sum()

classes_folders = ['fighter_jet', 'motorbike', 'school_bus', 'touring_bike', 'airplane', 'car_side']

ind = svm_classes.index.tolist()
for i in range(len(ind)):
    ind[i] = classes_folders[i]

svm_classes.index = ind

svm_classes.loc['average', 'prediction_correct'] = svm_classes['prediction_correct'].sum()
svm_classes.loc['average', 'num_images'] = svm_classes['num_images'].sum()
svm_classes['accuracy'] = svm_classes['prediction_correct'] / svm_classes['num_images']

svm_classes

plot_bar_x(svm_classes.index, svm_classes.accuracy.values, 'Class', 'Accuracy for each class')


svm_words = svm_grouped.groupby('vocabulary_words').sum()

svm_words['accuracy'] = svm_words['prediction_correct'] / svm_words['num_images']

svm_words

plot_bar_x(svm_words.index, svm_words.accuracy.values, 'Number of vocabulary words', 'Accuracy for different number of words in BoVW vocabulary')


svm_norms = svm_grouped.groupby('normalization').sum()

svm_norms['accuracy'] = svm_norms['prediction_correct'] / svm_norms['num_images']

svm_norms

plot_bar_x(svm_norms.index, svm_norms.accuracy.values, 'Normalization', 'Accuracy with and without normalization')


svm_kernel = svm_grouped.groupby('kernel').sum()

svm_kernel['accuracy'] = svm_kernel['prediction_correct'] / svm_kernel['num_images']

svm_kernel

plot_bar_x(svm_kernel.index, svm_kernel.accuracy.values, 'Kernel', 'Accuracy for different kernels')


svm_eps = svm_grouped.groupby('epsilon').sum()

svm_eps['accuracy'] = svm_eps['prediction_correct'] / svm_eps['num_images']

svm_eps

plot_bar_x(svm_eps.index, svm_eps.accuracy.values, 'Epsilon', 'Accuracy for different values of termination epsilon')
