"""
Classes to model a Hill Climbing agent.
"""

import numpy as np


class HillClimbingAgent():
    def __init__(self, state_size, action_size, seed, policy='deterministic'):
        """Initialize an Agent object.

        Params
        ======
            model: model object
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): Random seed
            policy (str): Either 'deterministic' or 'stochastic'
        """
        np.random.seed(seed)
        self.action_size = action_size
        self.policy = policy
        self.weights = 1e-4 * np.random.randn(state_size, action_size)  # weights for simple linear policy: state_space x action_space

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

    def act(self, state):
        probs = self.forward(state)
        if self.policy == 'deterministic':
            return np.argmax(probs)
        elif self.policy == 'stochastic':
            return np.random.choice(self.action_size, p=probs)
