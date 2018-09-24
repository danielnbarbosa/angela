"""
Models used by Deep Q Network agent:
- TwoLayer: MLP with two hidden layers
- Dueling: Dueling network
- Conv3D: CNN with 3 Conv3d layers

DQN uses a local and a target network with the same underlying model.
These wrapper classes end with '2x':
- TwoLayer2x
- Dueling2x
- Conv3D2x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TwoLayer(nn.Module):
    """ MLP with two hidden layers."""

    def __init__(self, state_size, action_size, fc_units, seed=0):
        super(TwoLayer, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0], fc_units[1])
        self.output = nn.Linear(fc_units[1], action_size)

    def forward(self, x):
        #print('in:  {}'.format(x.shape))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        #print('out:  {}'.format(x.shape))
        return x


class Dueling(nn.Module):
    """ Dueling network with one shared layer and separate value and advantage layers."""

    def __init__(self, state_size, action_size, fc_units, seed=0):
        super(Dueling, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.fc_s = nn.Linear(state_size, fc_units[0])     # shared fc layer
        self.fc_v = nn.Linear(fc_units[0], fc_units[1])    # state fc layer
        self.out_v = nn.Linear(fc_units[1], 1)             # state output
        self.fc_a = nn.Linear(fc_units[0], fc_units[1])    # advantage fc layer
        self.out_a = nn.Linear(fc_units[1], action_size)   # advantage output

    def forward(self, x):
        #print('in:  {}'.format(x.shape))
        s = F.relu(self.fc_s(x))                # shared
        v = self.out_v(F.relu(self.fc_v(s)))    # state
        a = self.out_a(F.relu(self.fc_a(s)))    # advantage
        q = v + (a - a.mean())
        #print('out: {}'.format(q.shape))
        return q


class Conv3D(nn.Module):
    """
    CNN for learning from pixels.  Assumes 4 stacked 84x84 uint8 RGB frames.
    Total parameters: 5M
    """

    def __init__(self, action_size, seed=0):
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
        x = x.float() / 255                  # normalization
        #print('in:  {}'.format(x.shape))
        x = F.relu(self.bn1(self.conv1(x)))  # convolutions
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)            # flatten
        x = F.relu(self.fc(x))               # fully connected layer
        x = self.output(x)
        #print('out: {}'.format(x.shape))
        return x



# Initialize local and target network with identical initial weights.

class TwoLayer2x():
    def __init__(self, state_size, action_size, fc_units, seed=0):
        self.local = TwoLayer(state_size, action_size, fc_units, seed).to(device)
        self.target = TwoLayer(state_size, action_size, fc_units, seed).to(device)
        print(self.local)
        summary(self.local, (state_size,))

class Dueling2x():
    def __init__(self, state_size, action_size, fc_units, seed=0):
        self.local = Dueling(state_size, action_size, fc_units, seed).to(device)
        self.target = Dueling(state_size, action_size, fc_units, seed).to(device)
        print(self.local)
        summary(self.local, (state_size,))

class Conv3D2x():
    def __init__(self, state_size, action_size, seed=0):
        self.local = Conv3D(action_size, seed).to(device)
        self.target = Conv3D(action_size, seed).to(device)
        print(self.local)
        summary(self.local, (state_size))
