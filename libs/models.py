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
        torch.cuda.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.output = nn.Linear(fc2_units, action_size)

    def forward(self, x):
        """Build a network that maps state -> action values."""
        #print('in:  {}'.format(x.shape))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        #print('out:  {}'.format(x.shape))
        return x


class FourHiddenLayerNet(nn.Module):
    """ Classic DQN. """

    def __init__(self, state_size, action_size, fc1_units, fc2_units, fc3_units, fc4_units,seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(FourHiddenLayerNet, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, fc4_units)
        self.output = nn.Linear(fc4_units, action_size)

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
        torch.cuda.manual_seed(seed)
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


class Simple3DConvNet(nn.Module):
    """
    3D Convolutional Neural Network for learning from pixels.
    Assumes 4 stacked RGB frames with dimensions of 84x84.
    """

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (tuple): Shape of state input
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Simple3DConvNet, self).__init__()
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



##### Define QNets with two copies of the above architectures. #####

class TwoHiddenLayerQNet():
    def __init__(self, state_size, action_size, fc1_units, fc2_units, seed):
        """Initialize local and target network with identical initial weights."""
        self.local = TwoHiddenLayerNet(state_size, action_size, fc1_units, fc2_units, seed).to(device)
        self.target = TwoHiddenLayerNet(state_size, action_size, fc1_units, fc2_units, seed).to(device)
        print(self.local)
        summary(self.local, (state_size,))

class FourHiddenLayerQNet():
    def __init__(self, state_size, action_size, fc1_units, fc2_units, fc3_units, fc4_units, seed):
        """Initialize local and target network with identical initial weights."""
        self.local = FourHiddenLayerNet(state_size, action_size, fc1_units, fc2_units, fc3_units, fc4_units, seed).to(device)
        self.target = FourHiddenLayerNet(state_size, action_size, fc1_units, fc2_units, fc3_units, fc4_units, seed).to(device)
        print(self.local)
        summary(self.local, (state_size,))

class DuelingQNet():
    def __init__(self, state_size, action_size, fc1_units, fc2_units, seed):
        """Initialize local and target network with identical initial weights."""
        self.local = DuelingNet(state_size, action_size, fc1_units, fc2_units, seed).to(device)
        self.target = DuelingNet(state_size, action_size, fc1_units, fc2_units, seed).to(device)
        print(self.local)
        summary(self.local, (state_size,))

class Simple3DConvQNet():
    def __init__(self, state_size, action_size, seed):
        """Initialize local and target network with identical initial weights."""
        self.local = Simple3DConvNet(state_size, action_size, seed).to(device)
        self.target = Simple3DConvNet(state_size, action_size, seed).to(device)
        print(self.local)
        summary(self.local, (state_size))
