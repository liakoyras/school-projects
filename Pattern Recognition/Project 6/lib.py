import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TwoLayerNetwork(nn.Module):
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
    with torch.no_grad():
        preds = model(X)
        y_pred = preds.argmax(1)
        
        loss = loss_fn(preds, y).item()
        
        correct = (y_pred == y).sum()
        accuracy = correct/len(y)
        
    if verbose:
        print("test loss:", f"{loss:>7f}")
        print("test acc :", f"{100*accuracy:>3f}%")
        
    return loss, accuracy


