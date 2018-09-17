"""
Classes to model a Hill Climbing agent.
"""

import numpy as np


class HillClimbing():
    def __init__(self, state_size, action_size, seed,
                 noise_scale=1e-2,
                 use_adaptive_noise=True,
                 noise_scale_in=2,
                 noise_scale_out=2,
                 noise_min=1e-3,
                 noise_max=2,
                 policy='deterministic'):
        """Initialize an Agent object.

        Params
        ======
            model: model object
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): Random seed
            noise_scale (float): standard deviation of additive noise
            use_adaptive_noise (bool): whether to implement adaptive noise
            noise_scale_in (int): factor to reduce noise by
            noise_scale_out (int): factor to increase noise by, set to 1 for simmulated annealing
            noise_min (int): minimum noise_scale
            noise_max (int): maximum noise scale
            policy (str): Either 'deterministic' or 'stochastic'
        """
        np.random.seed(seed)
        self.action_size = action_size
        self.noise_scale = noise_scale
        self.use_adaptive_noise = use_adaptive_noise
        self.noise_scale_in = noise_scale_in
        self.noise_scale_out = noise_scale_out
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.policy = policy
        self.weights = 1e-4 * np.random.randn(state_size, action_size)  # weights for simple linear policy: state_space x action_space
        self.max_best_return = -np.Inf               # overall best return
        self.max_best_weights = self.weights        # overall best weights

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

    def learn(self, pop_noise, pop_return, pop_rewards):
        # determine who got the highest reward
        pop_best_return = pop_return.max()

        # compare best return from current population to global best return
        if pop_best_return >= self.max_best_return: # found better weights
            self.max_best_return = pop_best_return
            self.max_best_weights += self.noise_scale * pop_noise[pop_return.argmax()]
            self.noise_scale = max(self.noise_min, self.noise_scale / self.noise_scale_in) if self.use_adaptive_noise else self.noise_scale
        else: # did not find better weights
            self.noise_scale = min(self.noise_max, self.noise_scale * self.noise_scale_out) if self.use_adaptive_noise else self.noise_scale

        # consider the best rewards from the current population for calculating stats
        pop_best_rewards = pop_rewards[pop_return.argmax()]
        return pop_best_rewards, pop_best_return
