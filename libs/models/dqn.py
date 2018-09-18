"""
Models used by Deep Q Network agent:
- TwoLayer
- FourLayer
- Dueling
- Conv3D

- TwoLayer2x
- FourLayer2x
- Dueling2x
- Conv3D2x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##### Define various neural network architectures. #####


class TwoLayer(nn.Module):
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
        super(TwoLayer, self).__init__()
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


class FourLayer(nn.Module):
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
        super(FourLayer, self).__init__()
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


class Dueling(nn.Module):
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
        super(Dueling, self).__init__()
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


class Conv3D(nn.Module):
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

    def forward(self, x):
        x = x.float() / 255
        #print('in:  {}'.format(x.shape))
        x = F.relu(self.bn1(self.conv1(x)))  # convolutions
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)            # flatten
        x = F.relu(self.fc(x))               # fully connected layer
        x = self.output(x)
        #print('out: {}'.format(x.shape))
        return x


##### Define QNets with two copies of the above architectures. #####

class TwoLayer2x():
    def __init__(self, state_size, action_size, fc_units, seed):
        """Initialize local and target network with identical initial weights."""
        self.local = TwoLayer(state_size, action_size, fc_units, seed).to(device)
        self.target = TwoLayer(state_size, action_size, fc_units, seed).to(device)
        print(self.local)
        summary(self.local, (state_size,))

class FourLayer2x():
    def __init__(self, state_size, action_size, fc_units, seed):
        """Initialize local and target network with identical initial weights."""
        self.local = FourLayer(state_size, action_size, fc_units, seed).to(device)
        self.target = FourLayer(state_size, action_size, fc_units, seed).to(device)
        print(self.local)
        summary(self.local, (state_size,))

class Dueling2x():
    def __init__(self, state_size, action_size, fc_units, seed):
        """Initialize local and target network with identical initial weights."""
        self.local = Dueling(state_size, action_size, fc_units, seed).to(device)
        self.target = Dueling(state_size, action_size, fc_units, seed).to(device)
        print(self.local)
        summary(self.local, (state_size,))

class Conv3D2x():
    def __init__(self, state_size, action_size, seed):
        """Initialize local and target network with identical initial weights."""
        self.local = Conv3D(state_size, action_size, seed).to(device)
        self.target = Conv3D(state_size, action_size, seed).to(device)
        print(self.local)
        summary(self.local, (state_size))
