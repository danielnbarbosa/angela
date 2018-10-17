"""
Models used by Hill Climbing agent:
- SingleLayerPerceptron: no hidden layer, softmax activation
"""

import numpy as np


class SingleLayerPerceptron():
    """Simple SLP."""

    def __init__(self, state_size, action_size, seed=0):
        np.random.seed(seed)
        # simple linear policy: state_size x action_size
        self.weights = 1e-4 * np.random.randn(state_size, action_size)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))  # subtracting np.max(x) for numerical stability
        return e_x / e_x.sum()

    def forward(self, state):
        x = np.dot(state, self.weights)
        return self._softmax(x)
