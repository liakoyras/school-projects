import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances, rand_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import lib

rng = np.random.default_rng(seed=42) # random number generator

save_location = './' # image save location

### Import Data ###
column_names = ['area', 'perimeter', 'compactness', 'length',
                'width', 'asymmetry', 'groove', 'class']
data = pd.read_csv('seeds_dataset.txt', delimiter='\t',
                   names=column_names, index_col=False)

feature_names = column_names[:-1]
features = data[feature_names]

### Distance Matrices ###
# Euclidean distance
euclidean = pairwise_distances(features, metric='euclidean')

plt.imshow(euclidean, origin='upper', cmap='hot_r')
plt.colorbar()
plt.savefig(save_location+'euclidean_distances.png')

# Cosine distance
cosine = pairwise_distances(features, metric='cosine')

plt.imshow(cosine, origin='upper', cmap='hot_r')
plt.savefig(save_location+'cosine_distances.png')

"""
Silhouette Analysis
"""
# Calculate Sillhouette Coefficients 
ks = [i for i in range(2, 11)]

# Squared Euclidean Distance
sil_sqeuc = lib.silhouette_analysis(features, ks, 'sqeuclidean', rng)

# Plot coefficient for each k
fig, ax = plt.subplots(figsize=(9, 8))
fig.supylabel("Silhouette scores")
fig.supxlabel("k")
ax.plot(ks, sil_sqeuc)
ax.set_title("Squared Euclidean Distance", fontsize = 13)

plt.savefig(save_location+"silhouette.png")


### Normalize Data ###
scl = StandardScaler()
norm_features = scl.fit_transform(features)
norm_features = pd.DataFrame(norm_features, columns=column_names[:-1])

### Silhouette Coefficients for Normalized Data ###
# Cosine Distance
sil_norm_cos = lib.silhouette_analysis(norm_features, ks, 'cosine', rng)

# Squared Euclidean Distance
sil_norm_sqeuc = lib.silhouette_analysis(norm_features, ks, rng=rng)

# Plot coefficients for normalized data
fig, ax2 = plt.subplots(figsize=(9, 8))
ax2.plot(ks, sil_norm_cos)
ax2.set_title("Cosine Distance", fontsize = 13)
ax2.set_xlabel("k")
ax2.set_ylabel("Silhouette scores")

plt.savefig(save_location+"silhouette_norm_cosine.png")

# comparative plot
fig, ax3 = plt.subplots(1, 2, figsize=(15, 6))
fig.supylabel("Silhouette scores")
fig.supxlabel("k")
ax3[1].plot(ks, sil_norm_sqeuc)
ax3[1].set_title("Squared Euclidean Distance", fontsize = 13)
ax3[0].plot(ks, sil_norm_cos)
ax3[0].set_title("Cosine Distance", fontsize = 13)

plt.savefig(save_location+"silhouette_norm_compare.png")


"""
Rand Index - Euclidean KMeans
"""
### 1 run ###
predictions = lib.kmeans_cluster(features, 3, 'sqeuclidean', rng=rng)
print("Clustering predictions:")
print(predictions)
print()

# Match cluster labels to true classes
# We know that the samples are sorted by class label, so we will exploit
# this to find what class the k-means output corresponds to, by finding the
# value that appears the most in the predictions of each class.
# sklean's rand_score does not care about the prediction labels as long as
# samples are in the same cluster, so this step is just a demonstration
preds_corrected, c = lib.match_labels(data['class'], predictions)
for i in range(3):
    print("Class", i+1, "is cluster", c[i])

print()
print("The corrected prediction labels:")
print(preds_corrected)
print()

ri = rand_score(data['class'], predictions)
print("Rand Index:", ri)
ri_corr = rand_score(data['class'], preds_corrected)
print("Rand Index using corrected labels:", ri_corr)


### 1 Run - Normalized Data ###
preds_n = lib.kmeans_cluster(norm_features, 3, 'sqeuclidean', rng=rng)

ri_n = rand_score(data['class'], preds_n)
print("Rand Index (normalized data):", ri_n)
print()


### 5 Runs ###
ris = []
for _ in range(5):
    predictions = lib.kmeans_cluster(features, 3, 'sqeuclidean', rng=rng)
    ris.append(rand_score(data['class'], predictions))

print("Rand Index for each run:", ris)
print("mean:", np.mean(ris), "std:", np.std(ris))

### 5 Runs - Normalized Data ###
ris_n = []
for _ in range(5):
    preds_n = lib.kmeans_cluster(features, 3, 'sqeuclidean', rng=rng)
    ris_n.append(rand_score(data['class'], preds_n))

print("Rand Index for each run (normalized data):", ris_n)
print("mean:", np.mean(ris_n), "std:", np.std(ris_n))
print()
print()


"""
K-means with cosine similarity
"""
print("K-Means with cosine distance")

# sklearn does not support different distance metrics, so pyclustering will be used for these steps
### 1 Run ###
predictions_cos = lib.kmeans_cluster(features, 3, 'cosine', rng=rng)
print("Clustering Predictions:")
print(predictions_cos)

ri_cos = rand_score(data['class'], predictions_cos)
print("Rand Index:", ri_cos)


### 1 Run - Normalized Data ###
preds_cos_n = lib.kmeans_cluster(norm_features, 3, 'cosine', rng=rng)
ri_cos_n = rand_score(data['class'], preds_cos_n)
print("Rand Index (normalized data):", ri_cos_n)
print()


### 5 Runs ##
ris_cos = []
for _ in range(5):
    predictions_cos = lib.kmeans_cluster(features, 3, 'cosine', rng=rng)
    ris_cos.append(rand_score(data['class'], predictions_cos))

print("Rand Index for each run:", ris_cos)
print("mean:", np.mean(ris_cos), "std:", np.std(ris_cos))


### 5 Runs - Normalized Data ##
ris_cos_n = []
for _ in range(5):
    preds_cos_n = lib.kmeans_cluster(norm_features, 3, 'cosine', rng=rng)
    ris_cos_n.append(rand_score(data['class'], preds_cos_n))

print("Rand Index for each run (normalized data):", ris_cos_n)
print("mean:", np.mean(ris_cos_n), "std:", np.std(ris_cos_n))
print()


"""
Best Model Analysis
"""
best_predictions = preds_n.copy()
print("The predictions for the best clusterer:")
print(best_predictions)

print("Corrected:")
best_preds_corr, _ = lib.match_labels(data['class'], best_predictions)
print(best_preds_corr)
print()

### Per Class Error ###
per_class_accuracies = confusion_matrix(data['class'],
                                        best_preds_corr,
                                        normalize='true'
                                        ).diagonal()

per_class_errors = 1 - per_class_accuracies
for count, error in enumerate(per_class_errors):
    print("Class", count+1, "error:", error)


