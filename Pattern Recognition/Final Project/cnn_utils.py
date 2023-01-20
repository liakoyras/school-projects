"""
Utilities that help with training and testing the neural networks.
"""
import torch

from torch import max as tmax

import matplotlib.pyplot as plt

def batch_train(model, images, labels, loss_fn, optimizer):
    """
    Basic neural network training functionality for a mini-batch.
    
    Parameters
    ----------
    model : class inheriting torch.nn.Module
        The model to be used for training.
    images : torch.Tensor of size batch_size x image_dims)
        A Tensor containing the batch data (returned from iterating through a
        DataLoader).
    labels : torch.Tensor of size batch_size
        A Tensor containing the batch labels (returned from iterating through a
        DataLoader).
    loss_fn : torch.nn loss function (or callable)
        The loss function that calculates the cost and does backpropagation.
    optimizer : torch.optim.Optimizer
        The optimizer that will be used to try and converge.
        
    Returns
    -------
    float
        The value of the loss function for this batch.
    """
    model.train()                   # train mode
    optimizer.zero_grad()           # zero gradient on optimizer
    
    outputs = model(images)         # feed forwared
    loss = loss_fn(outputs, labels) # calculate loss
    loss.backward()                 # backpropagation
    optimizer.step()                # update parameters based on backprop
    
    return loss.item()
    
def train_epoch(model, loss_fn, optimizer, train_loader, verbose=1, epoch=None):
    """
    Train a neural network for an epoch.
    
    Essentially iterate through all of the batches using the loader and call
    batch_train for all of them.
    
    Parameters
    ----------
    model : class inheriting torch.nn.Module
        The model to be used for training.
    loss_fn : torch.nn loss function
        The loss function that calculates the cost and does backpropagation.
    optimizer : torch.optim.Optimizer
        The optimizer that will be used to try and converge.
    train_loader : torch.utils.data.DataLoader
        The loader that iterates through batches of data.
    verbose : int
        2 is for printing information on every iteration/batch
        any other value is for no printable output
    epoch : int, optional
        This value is used when printing output to specify the overall epoch
        number that this call of the function represents.
        
    Returns
    -------
    list of float
        The values of the loss function for all batches in this epoch.
        Since the training happens sequentially, the last value should be the
        loss at the end of the epoch (after the final batch).
    """
    if verbose == 2:
        batches = len(train_loader)
    
    losses = []
    for i, (images, labels) in enumerate(train_loader):
        loss = batch_train(model, images, labels, loss_fn, optimizer)
        losses.append(loss)

        if verbose == 2 and epoch is not None:
            print(f"Epoch: {epoch+1}/{epochs}, Iter: {i+1}/{batches}, Loss: {loss:.4f}")
        elif verbose == 2 and epoch is None:
            print(f"Iter: {i+1}/{batches}, Loss: {loss:.4f}")
                 
    return losses

def test_model(model, test_loader):
    """
    Test the accuracy of a neural network.
    
    Make predictions and calculate the test accuray, defined as the number
    of correct predictions divided by the total number of test samples.
    
    The "mini-batch" approach is only for memory management reasons.
    
    Parameters
    ----------
    model : class inheriting torch.nn.Module
        The model to be tested.
    train_loader : torch.utils.data.DataLoader
        The loader that will
        
    Returns
    -------
    float
        The accuracy of the network (in this specific point in training).
    """
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)                  # make predictions
            _, predicted = tmax(outputs.data, 1)   # convert predictions
            total += labels.size(0)                # calculate number of images
            correct += (predicted == labels).sum() # calculate correct preds
    
    return correct/total

def train_loop(model, loss_fn, optimizer, train_loader, test_loader=None, epochs=50, verbose=1):
    """
    Train a neural network for a set number of epochs.
    
    It calls train_epoch as needed and also finds the model accuracy at
    the end of each epoch.
    
    Parameters
    ----------
    model : class inheriting torch.nn.Module
        The model to be used for training.
    loss_fn : torch.nn loss function
        The loss function that calculates the cost and does backpropagation.
    optimizer : torch.optim.Optimizer
        The optimizer that will be used to try and converge.
    train_loader : torch.utils.data.DataLoader
        The loader that iterates through batches of train data.
    test_loader : torch.utils.data.DataLoader, optional
        The loader that iterates through batches of test data.  
    epochs : int, default 50
        The total number of epochs that will run.
    verbose : int {1, 2, }
        2 is for printing the loss at every iteration/batch
        1 is for printing every epoch
        0 (and any other value) is for no printable output
        
    Returns
    -------
    losses : list of list of float
        The values of the loss function for all batches in all training epochs.
    accuracies : list of float
        The test accuracy for all training epochs.
    """
    losses = []
    accuracies = []
    for epoch in range(epochs):
        epoch_losses = train_epoch(model, loss_fn, optimizer, train_loader, verbose, epoch)
        losses.append(epoch_losses)
        
        if verbose == 1:
            print(f"Epoch: {epoch+1}/{epochs}, Loss: {epoch_losses[-1]:.4f}")
        
        if test_loader is not None:
            accuracy = test_model(model, test_loader)
            accuracies.append(accuracy)
            
            if verbose == 1 or verbose == 2:
                print(f"Epoch: {epoch+1}/{epochs}, Test Acc.: {accuracy*100:.2f}%")
        
    return losses, accuracies


def plot_loss(losses, accuracies=None, batches=False, percentage=True):
    """
    Plot the loss and test accuracy over a model's training.
    
    The function returns nothing, instead plt.show() is used to plot
    the graph directly on screen (without saving the image).
    
    Parameters
    ----------
    losses : list of list of float
        Each sublist contains the losses for each batch (like the output of
        train_loop)
    accuracies : list of float, optional
        The test accuracy for each epoch.
    batches : bool, default False
        Whether the created plot will contain the values for each batch (if
        False, epoch values will be used).
    percentage : bool, default True
        Whether or not to display the accuracy as percentages.
        
    See Also
    --------
    train_loop : Train a neural network for a set number of epochs.
    """
    if not batches:
        x_label = "Epoch"
        loss_x_axis = acc_x_axis = [i+1 for i in range(len(losses))]
        
        loss_y_axis = [epoch_losses[-1] for epoch_losses in losses]
        
    else:
        x_label = "Iteration"
        loss_x_axis = [i+1 for i in range(len(losses[0])*len(losses))]
        acc_x_axis = [(i+1)*len(losses[0]) for i in range(len(losses))]
        
        loss_y_axis = [loss for epoch_losses in losses for loss in epoch_losses]
        
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Loss")
    ax.plot(loss_x_axis, loss_y_axis, color='tab:red')
    
    if accuracies:
        if percentage:
            acc_y_axis = [acc*100 for acc in accuracies]
            acc_axis_name = "Accuracy (%)"
        else:
            acc_axis_name = "Accuracy"
            
        ax2 = ax.twinx()
        ax2.set_ylabel(acc_axis_name)
        ax2.plot(acc_x_axis, acc_y_axis, color='tab:blue')
    
    fig.legend(['Loss', 'Accuracy'])
    plt.show()
