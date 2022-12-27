import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
from torch import nn
from torch.optim import SGD

import lib

torch.manual_seed(42)
save_location = './'

# Import Data
col_names = ["s_len", "s_wid", "p_len", "p_wid", "class"]
data = pd.read_csv('iris.data', names=col_names)

# Replace class with numbers
# 0 - Iris Setosa
# 1 - Iris Versicolour
# 2 - Iris Virginica
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
data['class'] = data['class'].replace(class_names, [0, 1, 2])

print(data)

# Train-Test split
X = data.drop(['class'],axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.8,
                                                    random_state=42)

# Convert to torch Tensors
X_train = torch.from_numpy(X_train.values)
y_train = torch.from_numpy(y_train.values)
X_test = torch.from_numpy(X_test.values)
y_test = torch.from_numpy(y_test.values)


"""
Simple Two-Layer Neural Network
"""
dtype = X_train.dtype
tln = lib.TwoLayerNetwork(4, 3, 30, 'sigmoid', dtype=dtype)

cel = nn.CrossEntropyLoss()
sgd = SGD(tln.parameters(), lr=1e-3)

epochs = int(1e+5)
metrics = np.zeros((epochs, 3))
for epoch in range(epochs):
    v = not ((epoch+1) % 500)
    if v:
        print("Epoch " + str(epoch+1) + "/" + str(epochs))
        print(16*"--")

    train_loss = lib.train_nn(tln, X_train, y_train, cel, sgd, v)
    test_loss, test_acc = lib.test_nn(tln, X_test, y_test, cel, v)

    metrics[epoch] = [train_loss, test_loss, test_acc]
    if v:
        print(16*"--")

# Plot loss and accuracy over epochs
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(metrics[:,0], 'tab:blue')
ax1.plot(metrics[:,1], 'tab:orange')
ax2.plot(metrics[:,2], 'tab:green')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax2.set_ylabel('Accuracy')

plt.figlegend(['train loss', 'test loss', 'test accuracy'],
              loc='upper center', ncol=3)
plt.savefig(save_location+"two_layers.png")

# Confusion matrix
y_pred = tln(X_test).argmax(1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot()

plt.savefig(save_location+"two_layers_cm.png")

"""
With ReLU
"""
tln_relu = lib.TwoLayerNetwork(4, 3, 30, 'relu', dtype=dtype)

cel = nn.CrossEntropyLoss()
sgd = SGD(tln_relu.parameters(), lr=1e-3)

epochs = int(1e+5)
metrics_relu = np.zeros((epochs, 3))
for epoch in range(epochs):
    v = not ((epoch+1) % 500)
    if v:
        print("Epoch " + str(epoch+1) + "/" + str(epochs))
        print(16*"--")

    train_loss = lib.train_nn(tln_relu, X_train, y_train, cel, sgd, v)
    test_loss, test_acc = lib.test_nn(tln_relu, X_test, y_test, cel, v)

    metrics_relu[epoch] = [train_loss, test_loss, test_acc]
    if v:
        print(16*"--")

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(metrics_relu[:,0], 'tab:blue')
ax1.plot(metrics_relu[:,1], 'tab:orange')
ax2.plot(metrics_relu[:,2], 'tab:green')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax2.set_ylabel('Accuracy')

plt.figlegend(['train loss', 'test loss', 'test accuracy'],
              loc='upper center', ncol=3)
plt.savefig(save_location+"two_layers_relu.png")

# Confusion matrix
y_pred = tln_relu(X_test).argmax(1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot()

plt.savefig(save_location+"two_layers_relu_cm.png")


"""
Optimization
"""
# Different hidden layer neuron count
tln_60 = lib.TwoLayerNetwork(4, 3, 60, 'relu', dtype=dtype)
tln_15 = lib.TwoLayerNetwork(4, 3, 15, 'relu', dtype=dtype)

cel_1 = nn.CrossEntropyLoss()
cel_2 = nn.CrossEntropyLoss()

sgd_60 = SGD(tln_60.parameters(), lr=1e-3)
sgd_15 = SGD(tln_15.parameters(), lr=1e-3)

epochs = int(5e+3)
accuracies = np.zeros((epochs, 2))
for epoch in range(epochs):
    v = not ((epoch+1) % 500)
    if v:
        print("Epoch " + str(epoch+1) + "/" + str(epochs))
        print(16*"--")
        print("Net 1:")
    _ = lib.train_nn(tln_60, X_train, y_train, cel_1, sgd_60, v)
    _, test_acc_1 = lib.test_nn(tln_60, X_test, y_test, cel_1, v)
    accuracies[epoch, 0] = test_acc_1
    
    if v:
        print("Net 2:")
    _ = lib.train_nn(tln_15, X_train, y_train, cel_2, sgd_15, v)
    _, test_acc_2 = lib.test_nn(tln_15, X_test, y_test, cel_2, v)
    accuracies[epoch, 1] = test_acc_2
    
    if v:
        print(16*"--")


fig, ax1 = plt.subplots()

ax1.plot(accuracies[:,0], 'tab:blue')
ax1.plot(accuracies[:,1], 'tab:orange')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')

plt.figlegend(['60 neurons', '15 neurons'],
              loc='upper center', ncol=2)
plt.savefig(save_location+"optimization_5k.png")


# Changing learning rate
lrs = [1e-9, 1e-7, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
fig, ax1 = plt.subplots()

for count, lr in enumerate(lrs):
    tln_60 = lib.TwoLayerNetwork(4, 3, 60, 'relu', dtype=dtype)
    cel = nn.CrossEntropyLoss()
    sgd = SGD(tln_60.parameters(), lr=lr)
    epochs = int(5e+3)
    accuracies = np.zeros((epochs, len(lrs)))
    print("Learning rate:", lr)
    for epoch in range(epochs):
        v = not ((epoch+1) % 500)
        if v:
            print("Epoch " + str(epoch+1) + "/" + str(epochs))
            print(16*"--")
        _ = lib.train_nn(tln_60, X_train, y_train, cel, sgd, v)
        _, test_acc = lib.test_nn(tln_60, X_test, y_test, cel, v)
        accuracies[epoch, count] = test_acc
        if v:
            print(16*"--")
    
    ax1.plot(accuracies[:, count])
    ax1.legend(lrs, loc='lower right', ncol=3)

plt.savefig(save_location+"optimization_lrs_5k.png")

