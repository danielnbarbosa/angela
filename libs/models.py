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
