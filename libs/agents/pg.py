"""
Class to model a Policy Gradient agent.
"""

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PolicyGradient():
    def __init__(self, model, seed, lr=1e-2, action_map=None):
        """Initialize an Agent object.

        Params
        ======
            model: model object
            seed (int): Random seed
            lr (float): learning rate
            action_map (dict): how map action indexes from model to gym environment
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.model = model
        self.action_map = action_map
        self.optimizer = optim.Adam(model.parameters(), lr=lr)


    def _discount(self, rewards, gamma, normal):
        """Calulate discounted (and optionally normalized) rewards.
           From https://github.com/wagonhelm/Deep-Policy-Gradient
        """
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
        """Returns action and log probability for given state."""
        if len(state.shape) == 1:   # reshape 1-D states into 2-D (as expected by the model)
            state = np.expand_dims(state, axis=0)
        state = torch.from_numpy(state).float().to(device)
        probs = self.model.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        # if an action_map is defined then use it
        if self.action_map:
            return self.action_map[action.item()], m.log_prob(action)
        else:
            return action.item(), m.log_prob(action)

    def learn(self, rewards, saved_log_probs, gamma):
        """Update model weights."""
        # calculate discounted rewards for each step and normalize them
        discounted_rewards = self._discount(rewards, gamma, True)

        policy_loss = []
        for i, log_prob in enumerate(saved_log_probs):
            policy_loss.append(-log_prob * discounted_rewards[i])
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
