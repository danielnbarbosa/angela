"""
Hill Climbing agent.
"""

import pickle
import numpy as np


class HillClimbing():
    def __init__(self, model, action_size, seed, load_file=None,
                 noise_scale=1e-2,
                 use_adaptive_noise=True,
                 noise_scale_in=2,
                 noise_scale_out=2,
                 noise_min=1e-3,
                 noise_max=2,
                 policy='deterministic'):
        """
        Params
        ======
            model: model object
            action_size (int): dimension of each action
            seed (int): Random seed
            load_file (str): path of checkpoint file to load
            noise_scale (float): standard deviation of additive noise
            use_adaptive_noise (bool): whether to implement adaptive noise
            noise_scale_in (int): factor to reduce noise by
            noise_scale_out (int): factor to increase noise by, set to 1 for simmulated annealing
            noise_min (int): minimum noise_scale
            noise_max (int): maximum noise scale
            policy (str): Either 'deterministic' or 'stochastic'
        """
        np.random.seed(seed)
        self.model = model
        if load_file:
            model.weights = pickle.load(open(load_file, 'rb'))
            print('Loaded: {}'.format(load_file))
        self.action_size = action_size
        self.noise_scale = noise_scale
        self.use_adaptive_noise = use_adaptive_noise
        self.noise_scale_in = noise_scale_in
        self.noise_scale_out = noise_scale_out
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.policy = policy
        self.max_best_return = -np.Inf               # overall best return
        self.max_best_weights = model.weights        # overall best weights


    def act(self, state):
        """Given a state, determine the next action."""

        probs = self.model.forward(state)
        if self.policy == 'deterministic':
            return np.argmax(probs)
        elif self.policy == 'stochastic':
            return np.random.choice(self.action_size, p=probs)


    def learn(self, pop_noise, pop_return, pop_rewards):
        """If a better performing model was found then use it's weights."""

        # determine who got the highest reward
        pop_best_return = pop_return.max()

        # compare best return from current population to overall best return
        if pop_best_return >= self.max_best_return: # found better weights
            self.max_best_return = pop_best_return
            self.max_best_weights += self.noise_scale * pop_noise[pop_return.argmax()]
            self.noise_scale = max(self.noise_min, self.noise_scale / self.noise_scale_in) if self.use_adaptive_noise else self.noise_scale
        else: # did not find better weights
            self.noise_scale = min(self.noise_max, self.noise_scale * self.noise_scale_out) if self.use_adaptive_noise else self.noise_scale

        # consider the best rewards from the current population for calculating stats
        pop_best_rewards = pop_rewards[pop_return.argmax()]
        return pop_best_rewards, pop_best_return
