"""
Models used by Vanilla Policy Gradients agent:
- SingleHiddenLayer: simple multi-layer perceptron
- BigConv2D: CNN with 3 Conv2d layers
- Conv3D: CNN with 3 Conv3d layers
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
    CNN for learning from pixels.  Assumes 4 stacked 80x80 grayscale frames.
    Defaults to float32 frames.  If using uint8 frames, set normalize=True
    Total parameters: 243K
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
        self.fc = nn.Linear(filter_maps[1]*conv_out**2, fc_units)
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


class BigConv2D(nn.Module):
    """
    CNN for learning from pixels.  Assumes 4 stacked 80x80 float32 grayscale frames.
    Modeled after CNN architecture in orignial DQN Atari paper.
    Total parameters: 1.2M
    """

    def __init__(self, state_size, action_size, fc_units, seed=0):
        super(BigConv2D, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # input shape: (m, 4, 80, 80)                    shape after
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)       # (m, 16, 19, 19)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)      # (m, 32, 8, 8)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)      # (m, 32, 6, 6)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64*6*6, fc_units)            # (m, 2304, fc_units)
        self.output = nn.Linear(fc_units, action_size)   # (m, fc_units, n_a)
        print(self)  # print model
        summary(self.to(device), state_size)

    def forward(self, x):
        #print('in:  {}'.format(x))
        x = F.relu(self.bn1(self.conv1(x)))   # convolutions
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc(x))                # fully connected layer
        x = self.output(x)
        #print('out: {}'.format(x.shape))
        return F.softmax(x, dim=1)


class Conv3D(nn.Module):
    """
    CNN for learning from pixels.  Assumes 4 stacked 84x84 uint8 RGB frames.
    Total parameters: 5M
    """

    def __init__(self, state_size, action_size, seed=0):
        super(Conv3D, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # input shape: (m, 3, 4, 84, 84)                                shape after
        self.conv1 = nn.Conv3d(3, 128, (1, 3, 3), stride=(1, 3, 3))     # (m, 128, 4, 28, 28)
        self.bn1 = nn.BatchNorm3d(128)
        self.conv2 = nn.Conv3d(128, 256, (1, 3, 3), stride=(1, 3, 3))   # (m, 256, 4, 9, 9)
        self.bn2 = nn.BatchNorm3d(256)
        self.conv3 = nn.Conv3d(256, 256, (4, 3, 3), stride=(1, 3, 3))   # (m, 256, 1, 3, 3)
        self.bn3 = nn.BatchNorm3d(256)
        self.fc = nn.Linear(256*1*3*3, 1024)                            # (m, 2304, 1024)
        self.output = nn.Linear(1024, action_size)                      # (m, 512, n_a)
        print(self)  # print model
        summary(self.to(device), state_size)

    def forward(self, x):
        x = x.float() / 255                  # normalization
        #print('in:  {}'.format(x.shape))
        x = F.relu(self.bn1(self.conv1(x)))  # convolutions
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)            # flatten
        x = F.relu(self.fc(x))               # fully connected layer
        x = self.output(x)
        #print('out: {}'.format(x.shape))
        return F.softmax(x, dim=1)
