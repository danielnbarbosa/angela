"""
Class to model a Policy Gradient agent.
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
        # TODO don't do the summary here, should be in the model
        #summary(self.model, (state_size,))
        summary(self.model, state_size)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    # this function is from https://github.com/wagonhelm/Deep-Policy-Gradient
    def _discount(self, rewards, gamma, normal):
        discounted_rewards = np.zeros_like(rewards)
        G = 0.0
        for i in reversed(range(0, len(rewards))):
            G = G * gamma + rewards[i]
            discounted_rewards[i] = G
        # Normalize
        if normal:
            mean = np.mean(discounted_rewards)
            std = np.std(discounted_rewards)
            discounted_rewards = (discounted_rewards - mean) / (std)
        return discounted_rewards

    def act(self, state):
        # TODO make this work for both conv2d and flat networks
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        state = torch.from_numpy(state).float().to(device)
        probs = self.model.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        # TODO make this generic
        action_map = {0: 0, 1: 2, 2: 5}
        return action_map[action.item()], m.log_prob(action)


    def learn(self, rewards, saved_log_probs, gamma):
        # original code just calculates return from initial step
        #discounts = [gamma**i for i in range(len(rewards))]
        #R = sum([a*b for a,b in zip(discounts, rewards)])

        # same as above but uses numpy instead for better speed
        #discounts = gamma**np.arange(len(rewards))
        #R = np.dot(discounts, rewards)

        # calculating discounted rewards for each step and normalizing them
        # this made huge improvement in performance!
        discounted_rewards = self._discount(rewards, gamma, True)

        policy_loss = []
        for i, log_prob in enumerate(saved_log_probs):
            policy_loss.append(-log_prob * discounted_rewards[i])
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
