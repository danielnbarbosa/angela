import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicQNetwork(nn.Module):
    """ Classic DQN. """

    def __init__(self, state_size, action_size, fc1_units, fc2_units, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(ClassicQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.output = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x



class DuelingQNetwork(nn.Module):
    """ Dueling DQN. """

    def __init__(self, state_size, action_size, fc1_units, fc2_units, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc_s = nn.Linear(state_size, fc1_units)     # shared fc layer
        self.fc_v = nn.Linear(fc1_units, fc2_units)      # state fc layer
        self.out_v = nn.Linear(fc2_units, 1)             # state output
        self.fc_a = nn.Linear(fc1_units, fc2_units)      # advantage fc layer
        self.out_a = nn.Linear(fc2_units, action_size)   # advantage output

    def forward(self, state):
        """Build a network that maps state -> action values."""
        s = F.relu(self.fc_s(state))            # shared
        v = self.out_v(F.relu(self.fc_v(s)))    # state
        a = self.out_a(F.relu(self.fc_a(s)))    # advantage
        q = v + (a - a.mean())
        return q



class ConvNet(nn.Module):
    """Convolutional Neural Network for learning from pixels."""

    def __init__(self, action_size, seed):
        super(ConvNet, self).__init__()
        self.seed = torch.manual_seed(seed)

        # formula for calculcating conv net output dims: (W-F)/S + 1

        # image is converted to luminescence (grayscale) before arriving here
        # input shape: (1, 1, 84, 84)
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4)
        # new shape: (1, 32, 20, 20)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        # new shape: (1, 64, 9, 9)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        # new shape: (1, 64, 7, 7)
        self.fc = nn.Linear(64*7*7, 512)
        self.output = nn.Linear(512, action_size)

    def forward(self, x):
        # convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # flatten
        x = x.view(x.size(0), -1)
        # fully connected layer
        x = F.relu(self.fc(x))
        x = self.output(x)
        return x
