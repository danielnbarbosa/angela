"""
Classes to model a Policy Gradient agent.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PolicyGradientAgent():
    def __init__(self, model, state_size, seed,
                 lr=1e-2):
        """Initialize an Agent object.

        Params
        ======
            model: model object
            state_size (int): dimension of each state
            seed (int): Random seed
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.model = model
        print(self.model)
        summary(self.model, (state_size,))
        self.optimizer = optim.Adam(model.parameters(), lr=lr)


    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.model.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


    def learn(self, rewards, saved_log_probs, gamma):
        discounts = gamma**np.arange(len(rewards))
        R = np.dot(discounts, rewards)
        #discounts = [gamma**i for i in range(len(rewards))]
        #R = sum([a*b for a,b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
