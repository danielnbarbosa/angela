"""
Classes to model agent's neural networks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##### Define various neural network architectures. #####


class DQNTwoHiddenLayer(nn.Module):
    """ Classic DQN. """

    def __init__(self, state_size, action_size, fc_units, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc_units (tuple): Dimension of each hidden layer
            seed (int): Random seed
        """
        super(DQNTwoHiddenLayer, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0], fc_units[1])
        self.output = nn.Linear(fc_units[1], action_size)

    def forward(self, x):
        """Build a network that maps state -> action values."""
        #print('in:  {}'.format(x.shape))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        #print('out:  {}'.format(x.shape))
        return x


class DQNFourHiddenLayer(nn.Module):
    """ Classic DQN. """

    def __init__(self, state_size, action_size, fc_units, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc_units (tuple): Dimension of each hidden layer
            seed (int): Random seed
        """
        super(DQNFourHiddenLayer, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0], fc_units[1])
        self.fc3 = nn.Linear(fc_units[1], fc_units[2])
        self.fc4 = nn.Linear(fc_units[2], fc_units[3])
        self.output = nn.Linear(fc_units[3], action_size)

    def forward(self, x):
        """Build a network that maps state -> action values."""
        #print('in:  {}'.format(x.shape))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.output(x)
        #print('out:  {}'.format(x.shape))
        return x


class DQNDueling(nn.Module):
    """ Dueling DQN. """

    def __init__(self, state_size, action_size, fc_units, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc_units (tuple): Dimension of each hidden layer
            seed (int): Random seed
        """
        super(DQNDueling, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.fc_s = nn.Linear(state_size, fc_units[0])     # shared fc layer
        self.fc_v = nn.Linear(fc_units[0], fc_units[1])    # state fc layer
        self.out_v = nn.Linear(fc_units[1], 1)             # state output
        self.fc_a = nn.Linear(fc_units[0], fc_units[1])    # advantage fc layer
        self.out_a = nn.Linear(fc_units[1], action_size)   # advantage output

    def forward(self, x):
        """Build a network that maps state -> action values."""
        #print('in:  {}'.format(x.shape))
        s = F.relu(self.fc_s(x))                # shared
        v = self.out_v(F.relu(self.fc_v(s)))    # state
        a = self.out_a(F.relu(self.fc_a(s)))    # advantage
        q = v + (a - a.mean())
        #print('out: {}'.format(q.shape))
        return q


class DQNConv3D(nn.Module):
    """
    3D Convolutional Neural Network for learning from pixels using DQN.
    Assumes 4 stacked RGB frames with dimensions of 84x84.
    """

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DQNConv3D, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # formula for calculcating conv net output dims: (W-F)/S + 1
        # DQN Paper:
        # conv1: 32, 8x8, 4
        # conv2: 64, 4x4, 2
        # conv3: 64, 3x3, 1
        # fc: 512
        # input shape: (m, 3, 4, 84, 84)                                shape after
        self.conv1 = nn.Conv3d(3, 128, (1, 3, 3), stride=(1, 3, 3))     # (m, 128, 4, 28, 28)
        self.bn1 = nn.BatchNorm3d(128)
        self.conv2 = nn.Conv3d(128, 256, (1, 3, 3), stride=(1, 3, 3))   # (m, 256, 4, 9, 9)
        self.bn2 = nn.BatchNorm3d(256)
        self.conv3 = nn.Conv3d(256, 256, (4, 3, 3), stride=(1, 3, 3))   # (m, 256, 1, 3, 3)
        self.bn3 = nn.BatchNorm3d(256)
        self.fc = nn.Linear(256*1*3*3, 1024)                            # (m, 2304, 1024)
        self.output = nn.Linear(1024, action_size)                      # (m, 512, n_a)

    def forward(self, x):
        x = x.float() / 255
        #print('in:  {}'.format(x.shape))
        # convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # flatten
        x = x.view(x.size(0), -1)
        # fully connected layer
        x = F.relu(self.fc(x))
        x = self.output(x)
        #print('out: {}'.format(x.shape))
        return x


class HillClimbing():
    def __init__(self, state_size, action_size, seed):
        np.random.seed(seed)
         # weights for simple linear policy: state_space x action_space
        self.weights = 1e-4 * np.random.randn(state_size, action_size)

    def _softmax(self, x):
        exp = np.exp(x)
        return exp/np.sum(exp)

    def _softmax_stable(self, x):
        #exp = np.exp(x) - np.exp(max(x))
        #return exp/np.sum(exp)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def forward(self, state):
        x = np.dot(state, self.weights)
        #return np.exp(x)/sum(np.exp(x))
        return self._softmax_stable(x)


class PGOneHiddenLayer(nn.Module):
    def __init__(self, state_size, action_size, fc1_units, seed):
        super(PGOneHiddenLayer, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.output = nn.Linear(fc1_units, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return F.softmax(x, dim=1)


class PGConv2D(nn.Module):
    """
    2D Convolutional Neural Network for learning from pixels using Policy Gradients.
    Assumes 2 stacked greyscale frames with dimensions of 80x80.
    """

    def __init__(self, state_size, action_size, fc1_units, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (tuple): Shape of state input
            action_size (int): Dimension of each action
            fc1_units (int): Nodes in fully connected layer
            seed (int): Random seed
        """
        super(PGConv2D, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # formula for calculcating conv net output dims: (W-F)/S + 1
        # input shape: (m, 4, 80, 80)                     shape after
        self.conv1 = nn.Conv2d(4, 16, 8, stride=4)        # (m, 16, 19, 19)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=3)       # (m, 32, 6, 6)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32*6*6, fc1_units)            # (m, 800, fc1_units)
        self.output = nn.Linear(fc1_units, action_size)   # (m, fc1_units, n_a)
        # print model
        print(self)
        summary(self, state_size)

    def forward(self, x):
        #print('in:  {}'.format(x.shape))
        x = x.float() / 255
        #print('norm:  {}'.format(x.shape))
        # convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # flatten
        x = x.view(x.size(0), -1)
        # fully connected layer
        x = F.relu(self.fc(x))
        x = self.output(x)
        #print('out: {}'.format(x.shape))
        return F.softmax(x, dim=1)


class PGConv3D(nn.Module):
    """
    3D Convolutional Neural Network for learning from pixels using Policy Gradients.
    Assumes 4 stacked RGB frames with dimensions of 84x84.
    """

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(PGConv3D, self).__init__()
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
        # print model
        print(self)
        summary(self, state_size)

    def forward(self, x):
        x = x.float() / 255
        #print('in:  {}'.format(x.shape))
        # convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # flatten
        x = x.view(x.size(0), -1)
        # fully connected layer
        x = F.relu(self.fc(x))
        x = self.output(x)
        #print('out: {}'.format(x.shape))
        return F.softmax(x, dim=1)

##### Define QNets with two copies of the above architectures. #####

class DQNTwoHiddenLayer_Q():
    def __init__(self, state_size, action_size, fc_units, seed):
        """Initialize local and target network with identical initial weights."""
        self.local = DQNTwoHiddenLayer(state_size, action_size, fc_units, seed).to(device)
        self.target = DQNTwoHiddenLayer(state_size, action_size, fc_units, seed).to(device)
        print(self.local)
        summary(self.local, (state_size,))

class DQNFourHiddenLayer_Q():
    def __init__(self, state_size, action_size, fc_units, seed):
        """Initialize local and target network with identical initial weights."""
        self.local = DQNFourHiddenLayer(state_size, action_size, fc_units, seed).to(device)
        self.target = DQNFourHiddenLayer(state_size, action_size, fc_units, seed).to(device)
        print(self.local)
        summary(self.local, (state_size,))

class DQNDueling_Q():
    def __init__(self, state_size, action_size, fc_units, seed):
        """Initialize local and target network with identical initial weights."""
        self.local = DQNDueling(state_size, action_size, fc_units, seed).to(device)
        self.target = DQNDueling(state_size, action_size, fc_units, seed).to(device)
        print(self.local)
        summary(self.local, (state_size,))

class DQNConv3D_Q():
    def __init__(self, state_size, action_size, seed):
        """Initialize local and target network with identical initial weights."""
        self.local = DQNConv3D(state_size, action_size, seed).to(device)
        self.target = DQNConv3D(state_size, action_size, seed).to(device)
        print(self.local)
        summary(self.local, (state_size))
