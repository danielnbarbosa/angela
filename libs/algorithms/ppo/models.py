"""
Models used by Proximal Policy Optimization agent:
- SingleHiddenLayer: simple multi-layer perceptron
- TwoLayerConv2D: CNN with 2 Conv2d layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SingleHiddenLayer(nn.Module):
    """ MLP with one hidden layer."""

    def __init__(self, state_size, action_size, fc_units, seed=0):
        super(SingleHiddenLayer, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.output = nn.Linear(fc_units, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return F.softmax(x, dim=1)


class TwoLayerConv2D(nn.Module):
    """
    CNN for learning from pixels.  Assumes 4 stacked frames.
    Set normalize=True to convert grayscale (0 - 255) to floating point (0.0 - 1.0)
    """

    def __init__(self, state_size, action_size, filter_maps, kernels, strides, conv_out, fc_units, seed=0, normalize=False):
        """
        Params
        ======
        state_size: dimension of each state
        action_size (int): dimension of each action
        filter_maps (tuple): output size of each convolutional layer
        kernels (tuple): kernel size of each convolutional layer
        strides (tuple): stride size of each convolutional layer
        conv_out (int): size of final convolutional output
        fc_units (int): dimension of fully connected layer
        seed (int): random seed
        normalize (bool): whether to convert uint8 input to float32
        """
        super(TwoLayerConv2D, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.normalize = normalize
        # input shape: (m, 4, 80, 80)
        self.conv1 = nn.Conv2d(4, filter_maps[0], kernels[0], stride=strides[0])
        self.bn1 = nn.BatchNorm2d(filter_maps[0])
        self.conv2 = nn.Conv2d(filter_maps[0], filter_maps[1], kernels[1], stride=strides[1])
        self.bn2 = nn.BatchNorm2d(filter_maps[1])
        self.fc = nn.Linear(filter_maps[1]*conv_out[0]*conv_out[1], fc_units)
        self.output = nn.Linear(fc_units, action_size)
        print(self)  # print model
        summary(self.to(device), state_size)

    def forward(self, x):
        #print('in:  {}'.format(x))
        if self.normalize:                   # normalization
            x = x.float() / 255
        #print('norm:  {}'.format(x))
        x = F.relu(self.bn1(self.conv1(x)))  # convolutions
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)            # flatten
        x = F.relu(self.fc(x))               # fully connected layer
        x = self.output(x)
        #print('out: {}'.format(x))
        return F.softmax(x, dim=1)
