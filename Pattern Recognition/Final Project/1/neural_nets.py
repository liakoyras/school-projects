"""
Implementations of neural network architectures using PyTorch.
"""
import math
import torch
from torch import nn

class CNN(nn.Module):
    """
    Represent a parameterizeable CNN using PyTorch (inheriting from nn.Module).
    
    The architecture consists of several blocks of
    
    convolution + (maybe) batch normalization + activation + pooling 
    
    followed by flattening and fully connected layers.
    
    For the sake of simplicity, this class makes certain assumptions and does
    not attempt to make everything customizable.
    
    Note that it is possible to modify this architecture, for example by
    using pooling with 0 size kernel and no batch normalization in order to
    chain purely convolutional layers together.
    
    Things that CAN be parameterized:
    - Number of convolutional and fully connected layers
    - Number of kernels/neurons per layer
    - Kernel size, activation function, pooling kernel per layer
    - Batch normalization at each convolution layer
    
    (Some of the) Things that CAN'T be parameterized:
    - Convolution padding (output will always have the dimensions of input)
    - Stride, padding and dilation for convolution window (padding='same')
    - Pooling kernel shape, stride, padding and dilation (square, stride=size)
    - Wether or not the fully connected layers have bias
    - The position of activation and batch normalization in the pipeline
    
    Also, due to the calculation of the dimensions of the input to the fully
    connected layers, only square images are supported.
    
    Attributes
    ----------
    conv_layers : nn.ModuleList() of nn.Sequential()
        A list of Sequentials that each contains at least a conv layer,
        activation and max pooling, and possibly batch normalization.
    flatten : nn.Flatten()
        The object to convert data to 1-D before the fully connected layers.
    fc_layers : nn.ModuleList() of nn.Linear()
        A list of Linear nodes representing len-1 fully connected layers.
        
    Methods
    -------
    forward(self, x):
        Define how the network is run.
    """
    def __init__(self, n_conv_layers, filters, kernel, activation, norm, pool, input_channels, fully_connected, input_dims, classes):
        """
        Constructs all the necessary layers for the CNN object.

        Parameters
        ----------
        n_conv_layers : int
            The number of convolutional layers, used only to check against the
            lengths of the other parameters, as an extra safeguard.
        filters : list of int
            The number of convolution kernels at each layer.
        kernel : list of int
            The size of the (square) convolution kernel at each layer.
        activation : list of {'relu'}
            The activation functions after each convolution layer.
        norm : list of bool
            Whether or not each layer will have batch normalization.
        pool : list of int
            The size of the (square) pooling kernel at each layer. If 0 is
            passed, this pooling layer will be ignored.
        input_channels : int
            The number of channels that the input image has.
            For example, 1 for grayscale and 3 for RGB.
        fully_connected : list of int
            The number of neurons on each fully connected layer.            
        input_dims : tuple of int
            The dimensions (width, height) of input images. It will be used
            for the calculation of the size of the data after the flattening
            in order to create the fully connected layers with the right
            number of input parameters.
        classes : int
            The number of target classes. It will be used for the final output
            size.
        
        Raises
        ------
        ValueError
            If the lengths of filters, kernel, activation, pool and norm
            parameters are not equal to n_conv_layers.
        ValueError
            If the length of input_dims is not 2.
        ValueError
            If the dimensions in input_dims are not the same.
        """
        super(CNN, self).__init__()
        if not all(len(p) == n_conv_layers for p in [filters, kernel, activation, norm]):
            raise ValueError("The length of filters, kernel, activation, pool "\
                             "and norm parameters must be equal to the number "\
                             "of layers (n_conv_layers parameter).")
        if len(input_dims) == 2:
            if input_dims[0] != input_dims[1]:
                raise ValueError("Input image must be square.")
        else:
            raise ValueError("input_dims must be of length 2.")
        
        # Create convolutional layers (with activation, pooling etc)
        self.conv_layers = nn.ModuleList()
        for f, k, a, p, n in zip(filters, kernel, activation, pool, norm):
            layer = []
            layer.append(nn.Conv2d(input_channels, f, kernel_size=k, padding='same'))
            
            input_channels = f # input depth for the next layer
            
            if n: # batch norm before activation
                layer.append(nn.BatchNorm2d(f))
            
            if a == 'relu':
                layer.append(nn.ReLU())
            else:
                layer.append(nn.ReLU())
            
            if p != 0:
                layer.append(nn.MaxPool2d(p))
            
            layers = nn.Sequential(*layer)
            self.conv_layers.append(layers)
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Augment fully connected layers size list
        
        fully_connected.append(classes)
        # calculate size after each pooling
        dims = input_dims
        for p in pool:
            # formula adapted from 
            # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
            # using the assumptions of this class
            if p != 0:
                dims  = tuple( ((dim - (p-1) -1)/p) + 1  for dim in dims)
        
        flat_size = int(math.prod(tuple(dim  for dim in dims)) * filters[-1])
        fully_connected.insert(0, flat_size)
        
        # Create fully connected layers
        self.fc_layers = nn.ModuleList()
        for neur_in, neur_out in zip(fully_connected, fully_connected[1:]):
            fc_layer = nn.Linear(neur_in, neur_out)
            self.fc_layers.append(fc_layer)

    def forward(self, x):
        """
        Define how the network is run.
        
        Chain the convolutional layers at the beginning, then flatten and
        finally chain the fully connected layers.
        """
        if len(self.conv_layers) > 1:
            layer_out = self.conv_layers[0](x)
            for layer in self.conv_layers[1:]:
                layer_out = layer(layer_out)
        else:
            layer_out = self.conv_layers[0](x)
       
        flat = self.flatten(layer_out)
        
        if len(self.fc_layers) > 1:
            final_out = self.fc_layers[0](flat)
            for fc_layer in self.fc_layers[1:]:
                final_out = fc_layer(final_out)
        else:
            final_out = self.fc_layers[0](flat)
        
        return final_out



class FCN(nn.Module):
    """
    Represent a parameterizeable FCN using PyTorch (inheriting from nn.Module).
    
    The intention of this Fully Convolutional architecture is to mimic the
    behavior of a vanilla CNN that uses Fully Connected layers after the
    convolutional ones, while utilizing the input size invariability that a
    rolling convolution window offers as opposed to a linear transformation
    fully connected layer.
    
    Instead of fully connected layers, the final layers are convolutions with
    kernel size equal to the size of the image, along with global pooling
    (to achieve the afforementioned invariability).
    
    In addition, the convolutional layers + pooling are much less prone to
    overfitting and continue to keep the spatial relationships between the
    features, as opposed to the fully connected layers.
    
    For more information on the architecture and the assumptions this class
    makes, see the CNN class.
    
    Attributes
    ----------
    conv_layers : nn.ModuleList() of nn.Sequential()
        A list of Sequentials that each contains at least a conv layer,
        activation and max pooling, and possibly batch normalization.
        The object to convert data to 1-D before the fully connected layers.
    fcnn_layers : nn.ModuleList() of nn.Conv2d()
        A list of convolutional nodes that take the place of fully connected
        layers on a vanilla CNN.
        
    See Also
    -------
    CNN : Represent a parameterizeable CNN using PyTorch.
    """
    def __init__(self, n_conv_layers, filters, kernel, activation, norm, pool, input_channels, fully_convolutional, input_dims, classes):
        """
        Constructs all the necessary layers for the CNN object.
        
        All parameters and errors raised are the same as the CNN class, except
        fully_convolutional that replaces fully_connected and is explained
        below.

        Parameters
        ----------
        fully_convolutional : list of int
            The number of filters on each of the final fully convolutional
            layers. In case an instance is modeled after a CNN+fully connected
            architecture, this is equivalent to the number of neurons on each 
            of those final layers.
        """
        super(FCN, self).__init__()
        if not all(len(p) == n_conv_layers for p in [filters, kernel, activation, norm]):
            raise ValueError("The length of filters, kernel, activation, pool "\
                             "and norm parameters must be equal to the number "\
                             "of layers (n_conv_layers parameter).")
        if len(input_dims) == 2:
            if input_dims[0] != input_dims[1]:
                raise ValueError("Input image must be square.")
        else:
            raise ValueError("input_dims must be of length 2.")
        
        # Create convolutional layers (with activation, pooling etc)
        self.conv_layers = nn.ModuleList()
        for f, k, a, p, n in zip(filters, kernel, activation, pool, norm):
            layer = []
            layer.append(nn.Conv2d(input_channels, f, kernel_size=k, padding='same'))
                         
            input_channels = f # input depth for the next layer
            
            if n: # batch norm before activation
                layer.append(nn.BatchNorm2d(f))
            
            if a == 'relu':
                layer.append(nn.ReLU())
            else:
                layer.append(nn.ReLU())
            
            if p != 0:
                layer.append(nn.MaxPool2d(p))
            
            layers = nn.Sequential(*layer)
            self.conv_layers.append(layers)
        
        
        # Calculate size of image after the convolutional layers
        dims = input_dims
        for p in pool:
            # formula adapted from 
            # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
            if p != 0:
                dims  = tuple( int(((dim - (p-1) -1)/p) + 1) for dim in dims)
        
        # Create final fully convolutional layers
        self.fcnn_layers = nn.ModuleList()

        fcnn1 = nn.Conv2d(filters[-1], fully_convolutional[0], kernel_size=dims, padding=0)
        self.fcnn_layers.append(fcnn1)
        
        fully_convolutional.append(classes)
        for chan_in, chan_out in zip(fully_convolutional, fully_convolutional[1:]):
            fcnn_layer = nn.Conv2d(chan_in, chan_out, kernel_size=1, padding=0)
            self.fcnn_layers.append(fcnn_layer)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        """
        Define how the network is run.
        
        Chain the convolutional layers (augmented with any pooling etc.) at
        the beginning, and then the pure convolutions and global pooling.
        """
        if len(self.conv_layers) > 1:
            layer_out = self.conv_layers[0](x)
            for layer in self.conv_layers[1:]:
                layer_out = layer(layer_out)
        else:
            layer_out = self.conv_layers[0](x)
        
        if len(self.fcnn_layers) > 1:
            final_out = self.fcnn_layers[0](layer_out)
            for fcnn_layer in self.fcnn_layers[1:]:
                final_out = fcnn_layer(final_out)
        else:
            final_out = self.fcnn_layers[0](layer_out)
        
        pooled_out = self.gap(final_out)
        out = torch.squeeze(pooled_out)
        
        return out
