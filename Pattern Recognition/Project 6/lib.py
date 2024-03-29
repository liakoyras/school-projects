import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TwoLayerNetwork(nn.Module):
    """
    Wrapper class for PyTorch neural network with two layers.

    The architecture is:
    Input layer -> Hidden layer -> Output layer
    (all fully connected)
    """
    def __init__(self, in_dims, out_dims, hid_dims, activation, dtype=torch.float64):
        super(TwoLayerNetwork, self).__init__()
        self.layer1 = nn.Linear(in_dims, hid_dims, dtype=dtype)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ReLU()

        self.layer2 = nn.Linear(hid_dims, out_dims, dtype=dtype)

    def forward(self, x):
        out1 = self.activation(self.layer1(x))
        out2 = self.layer2(out1)

        return out2



def train_nn(model, X, y, loss_fn, optimizer, verbose=False):
    """
    Train a PyTorch neural network.
    
    This is essentially only one training epoch, for a complete training it
    needs t be wrapped in a loop.
    
    Parameters
    ----------
    model : pytorch nn.Module object
        The network to train.
    X : torch.Tensor 
        The train features.
    y : torch.Tensor
        The target variable.
    loss_fn : torch.nn loss function
        The loss function to be used for evaluation
    optimizer : torch.optim
        The optimizer algorithm to be used.
    verbose : bool, default True
        Wether or not the loss after this step should be printed.
    
    Returns
    -------
    float
        The loss at this epoch.
    """
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if verbose:
        print("train loss:", f"{loss:>7f}")
    
    return loss.item()

def test_nn(model, X, y, loss_fn, verbose=False):
    """
    Calculate a neural network's accuracy and loss on a test set.
    
    It uses
    Parameters
    ----------
    model : pytorch nn.Module object
        The trained network (at any training epoch).
    X : torch.Tensor
        The test set features.
    y : torch.Tensor
        The target variable.
    loss_fn : torch.nn loss function
        The loss function tο calculate.
    verbose : bool, default True
        Whether or not the loss and accuracy should be printed.
    
    Returns
    -------
    loss : float
        The test set loss.
    accuracy : float
        The test set accuracy.
    """
    with torch.no_grad(): # to avoid any gradient updating
        preds = model(X)
        y_pred = preds.argmax(1)
        
        loss = loss_fn(preds, y).item()
        
        correct = (y_pred == y).sum()
        accuracy = correct/len(y)
        
    if verbose:
        print("test loss:", f"{loss:>7f}")
        print("test acc :", f"{100*accuracy:>3f}%")
        
    return loss, accuracy

