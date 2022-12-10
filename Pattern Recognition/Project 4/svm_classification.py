import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

import lib

save_location = './'

### Import Data ###
column_names = ['area', 'perimeter', 'compactness', 'length',
                'width', 'asymmetry', 'groove', 'class']
data = pd.read_csv('seeds_dataset.txt', delimiter='\t',
                   names=column_names, index_col=False)
print(data)


### Train - Validation - Test split ###
train, non_train = train_test_split(data, train_size=120,
                                    stratify=data['class'], random_state=42)

val, test = train_test_split(non_train, test_size=0.5,
                             stratify=non_train['class'], random_state=42)

# confirm that the class ratio is consistent between the sets
for s in [train, val, test]:
    temp = []
    for c in [1, 2, 3]:
        temp.append(s[s['class'] == c].shape[0])
    
    if(max(temp) - min(temp) >= 2):
        print("Warning: The class ratios are not consistent in each set")
        print(temp)


print()

"""
SVM Binary Classification
"""
print("Binary Classification")

# Use only the two classes for this experiment
train_2 = train[train['class'] != 2]
val_2 = val[val['class'] != 2]
test_2 = test[test['class'] != 2]

feature_names = column_names[:-1]

X_train_2 = train_2[feature_names]
X_val_2 = val_2[feature_names]
X_test_2 = test_2[feature_names]

### Linear SVM ###
#### Tune C with validation set ###
Cs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e+0,
      1e+1, 1e+2, 1e+3, 1e+4, 1e+5, 1e+6,1e+9]
errors = []
models = []
for C in Cs:
    model = SVC(kernel='linear', C=C)

    model.fit(X_train_2, train_2['class'])

    predictions = model.predict(X_val_2)
    error = 1 - accuracy_score(val_2['class'], predictions)

    print("Using C =", C, " the validation error is", error)

    errors.append(error)
    models.append(model)

# Keep best model
best_lin = models[np.argmin(errors)]
print("The best model is for C =", Cs[np.argmin(errors)])
print()

### Test Data ###
test_predictions = best_lin.predict(X_test_2)
test_error = 1 - accuracy_score(test_2['class'], test_predictions)

print("The error on the test set is", test_error)
print()

### Predictions with Cross Validation ###
cv_errors = lib.cross_validation(best_lin, X_train_2, train_2['class'])

print("Cross Validation error for each fold:", cv_errors)
print("mean:", np.mean(cv_errors), "std:", np.std(cv_errors))
print()

### Non-linear Kernel SVM ###
kernels = ['poly', 'rbf', 'sigmoid']

# Find best kernel and tune C
errors = []
models = []
for k in kernels:
    C_errors = []
    C_models = []
    for C in Cs:
        model = SVC(kernel=k, C=C)
        
        model.fit(X_train_2, train_2['class'])
        
        predictions = model.predict(X_val_2)
        error = 1 - accuracy_score(val_2['class'], predictions)
        
        C_errors.append(error)
        C_models.append(model)
    
    print("The best", k, "kernel validation error is",
            np.min(C_errors), "for C =", Cs[np.argmin(C_errors)])
    errors.append(C_errors)
    models.append(C_models)


best_pol = models[0][np.argmin(errors[0])]
best_rbf = models[1][np.argmin(errors[1])]
best_sig = models[2][np.argmin(errors[2])]

print()

# Test set
pol_test_predictions = best_pol.predict(X_test_2)
pol_test_error = 1 - accuracy_score(test_2['class'], pol_test_predictions)
print("The test set error using polynomial kernel is", pol_test_error)

rbf_test_predictions = best_rbf.predict(X_test_2)
rbf_test_error = 1 - accuracy_score(test_2['class'], rbf_test_predictions)
print("The test set error using RBF kernel is", rbf_test_error)

sig_test_predictions = best_sig.predict(X_test_2)
sig_test_error = 1 - accuracy_score(test_2['class'], sig_test_predictions)
print("The test set error using sigmoid kernel is", sig_test_error)

print()

# Cross Validation
cv_errors_pol = lib.cross_validation(best_pol, X_train_2, train_2['class'])
cv_errors_rbf = lib.cross_validation(best_rbf, X_train_2, train_2['class'])
cv_errors_sig = lib.cross_validation(best_sig, X_train_2, train_2['class'])

print("Polynomial Kernel")
print("Cross Validation error for each fold:", cv_errors_pol)
print("mean:", np.mean(cv_errors_pol), "std:", np.std(cv_errors_pol))

print("RBF Kernel")
print("Cross Validation error for each fold:", cv_errors_rbf)
print("mean:", np.mean(cv_errors_rbf), "std:", np.std(cv_errors_rbf))

print("Sigmoid Kernel")
print("Cross Validation error for each fold:", cv_errors_sig)
print("mean:", np.mean(cv_errors_sig), "std:", np.std(cv_errors_sig))

print()
print()

"""
SVM Multi-class Classification
"""
print("3 Classes")

# variable shorthands
X_train = train[feature_names]
X_val = val[feature_names]
X_test = test[feature_names]

### Find best C for 3 classes ###
errors = []
for C in Cs:
    predictions, _ = lib.multiclass_svm(train, X_val, kernel='linear', C=C)
    error = 1 - accuracy_score(val['class'], predictions)

    print("Using C =", C, " the validation error is", error)
    errors.append(error)

best_C = Cs[np.argmin(errors)]
print("The best model is for C =", best_C)

### Predict Test Set ###
test_predictions_mc, _ = lib.multiclass_svm(train, X_test, C=best_C)
test_error_mc = 1 - accuracy_score(test['class'], test_predictions_mc)
print("The error on the test set is", test_error_mc)

### Confusion Matrix ###
ConfusionMatrixDisplay.from_predictions(test['class'], test_predictions_mc)

plt.savefig(save_location+"confusion_matrix.png")
