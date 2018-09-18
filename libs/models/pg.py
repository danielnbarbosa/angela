"""
Models used by Policy Gradients agent:
- SingleHiddenLayer
- SmallConv2D
- BigConv2D
- Conv3D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SingleHiddenLayer(nn.Module):
    def __init__(self, state_size, action_size, fc1_units, seed):
        super(SingleHiddenLayer, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.output = nn.Linear(fc1_units, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return F.softmax(x, dim=1)


class SmallConv2D(nn.Module):
    """
    2D Convolutional Neural Network for learning from pixels using Policy Gradients.
    Assumes 4 stacked greyscale frames with dimensions of 80x80.
    Total parameters: 243K
    """

    def __init__(self, state_size, action_size, fc_units, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (tuple): Shape of state input
            action_size (int): Dimension of each action
            fc1_units (int): Nodes in fully connected layer
            seed (int): Random seed
        """
        super(SmallConv2D, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # input shape: (m, 4, 80, 80)                    shape after
        self.conv1 = nn.Conv2d(4, 16, 8, stride=4)       # (m, 16, 19, 19)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=3)      # (m, 32, 6, 6)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32*6*6, fc_units)            # (m, 800, fc_units)
        self.output = nn.Linear(fc_units, action_size)   # (m, fc1_units, n_a)

        print(self)  # print model
        summary(self.to(device), state_size)

    def forward(self, x):
        #print('in:  {}'.format(x.shape))
        x = x.float() / 255
        #print('norm:  {}'.format(x.shape))
        x = F.relu(self.bn1(self.conv1(x)))  # convolutions
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)            # flatten
        x = F.relu(self.fc(x))               # fully connected layer
        x = self.output(x)
        #print('out: {}'.format(x.shape))
        return F.softmax(x, dim=1)


class BigConv2D(nn.Module):
    """
    2D Convolutional Neural Network for learning from pixels using Policy Gradients.
    Assumes 4 stacked greyscale frames with dimensions of 80x80.
    Modeled after CNN architecture in orignial DQN Atari paper.
    Total parameters: 1260K
    """

    def __init__(self, state_size, action_size, fc_units, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (tuple): Shape of state input
            action_size (int): Dimension of each action
            fc_units (int): Nodes in fully connected layer
            seed (int): Random seed
        """
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
        x = x.float() / 255
        #print('in:  {}'.format(x.shape))
        x = F.relu(self.bn1(self.conv1(x)))  # convolutions
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)            # flatten
        x = F.relu(self.fc(x))               # fully connected layer
        x = self.output(x)
        #print('out: {}'.format(x.shape))
        return F.softmax(x, dim=1)
