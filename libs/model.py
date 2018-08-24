"""
Classes to model agent's neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##### Define various neural network architectures. #####

class TwoHiddenLayerNet(nn.Module):
    """ Classic DQN. """

    def __init__(self, state_size, action_size, fc1_units, fc2_units, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(TwoHiddenLayerNet, self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.output = nn.Linear(fc2_units, action_size)

    def forward(self, x):
        """Build a network that maps state -> action values."""
        #print('in:  {}'.format(x.shape))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        #print('out: {}'.x(q.shape))
        return x



class DuelingNet(nn.Module):
    """ Dueling DQN. """

    def __init__(self, state_size, action_size, fc1_units, fc2_units, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingNet, self).__init__()
        torch.manual_seed(seed)
        self.fc_s = nn.Linear(state_size, fc1_units)     # shared fc layer
        self.fc_v = nn.Linear(fc1_units, fc2_units)      # state fc layer
        self.out_v = nn.Linear(fc2_units, 1)             # state output
        self.fc_a = nn.Linear(fc1_units, fc2_units)      # advantage fc layer
        self.out_a = nn.Linear(fc2_units, action_size)   # advantage output

    def forward(self, x):
        """Build a network that maps state -> action values."""
        #print('in:  {}'.format(x.shape))
        s = F.relu(self.fc_s(x))                # shared
        v = self.out_v(F.relu(self.fc_v(s)))    # state
        a = self.out_a(F.relu(self.fc_a(s)))    # advantage
        q = v + (a - a.mean())
        #print('out: {}'.format(q.shape))
        return q



class ConvNet(nn.Module):
    """
    Convolutional Neural Network for learning from pixels.
    Works with a variety of input channels: 1: greyscale, 3: RGB
    """

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (tuple): Shape of state input
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(ConvNet, self).__init__()
        torch.manual_seed(seed)
        # formula for calculcating conv net output dims: (W-F)/S + 1
        self.input_channels = state_size[0]  # number of color channels
        self.dim = state_size[1]             # length of one side of square image

        if self.dim == 84:
            # input shape: (m, input_channels, 84, 84)                      shape after
            self.conv1 = nn.Conv2d(self.input_channels, 32, 8, stride=4)    # (m, 32, 20, 20)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 4, stride=2)                     # (m, 64, 9, 9)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, 3, stride=2)                     # (m, 128, 4, 4)
            self.bn3 = nn.BatchNorm2d(128)
            self.fc = nn.Linear(128*4*4, 512)                                # (m, 2048, 512)
            self.output = nn.Linear(512, action_size)                       # (m, 512, n_a)

        elif self.dim == 42:
            # input shape: (m, input_channels, 42, 42)                      shape after
            self.conv1 = nn.Conv2d(self.input_channels, 32, 6, stride=4)    # (m, 32, 10, 10)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 2, stride=2)                     # (m, 64, 5, 5)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 64, 2, stride=1)                     # (m, 64, 4, 4)
            self.bn3 = nn.BatchNorm2d(64)
            self.fc = nn.Linear(64*4*4, 256)                                # (m, 1024, 256)
            self.output = nn.Linear(256, action_size)                       # (m, 256, n_a)


    def forward(self, x):
        #print('in:  {}'.format(x.shape))
        # reshape state output from environment to fit torch conv2d format (m, h, w, c) -> (m, c, h, w)
        x = x.reshape(-1, self.input_channels, self.dim, self.dim)
        #print('tx:  {}'.format(x.shape))
        # convolutions
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        # flatten
        x = x.view(x.size(0), -1)
        # fully connected layer
        x = F.elu(self.fc(x))
        x = self.output(x)
        #print('out: {}'.format(x.shape))
        return x



##### Define QNets with two copies of the above architectures. #####


class TwoHiddenLayerQNet():
    def __init__(self, state_size, action_size, fc1_units, fc2_units, seed):
        """Initialize local and target network with identical initial weights."""
        self.local = TwoHiddenLayerNet(state_size, action_size, fc1_units, fc2_units, seed).to(device)
        self.target = TwoHiddenLayerNet(state_size, action_size, fc1_units, fc2_units, seed).to(device)
        print(self.local)
        summary(self.local, (state_size,))

class DuelingQNet():
    def __init__(self, state_size, action_size, fc1_units, fc2_units, seed):
        """Initialize local and target network with identical initial weights."""
        self.local = DuelingNet(state_size, action_size, fc1_units, fc2_units, seed).to(device)
        self.target = DuelingNet(state_size, action_size, fc1_units, fc2_units, seed).to(device)
        print(self.local)
        summary(self.local, (state_size,))

class ConvQNet():
    def __init__(self, state_size, action_size, seed):
        """Initialize local and target network with identical initial weights."""
        self.local = ConvNet(state_size, action_size, seed).to(device)
        self.target = ConvNet(state_size, action_size, seed).to(device)
        print(self.local)
        summary(self.local, (state_size))
