import random
import numpy as np
from unityagents import UnityEnvironment
import gym
from discretize import create_uniform_grid
from skimage.color import rgb2gray

class Environment():
    """ Define an environment, currently either OpenAI Gym or UnityML. """

    def __init__(self, name, type, max_steps=None, one_hot=None, action_bins=None):
        self.type = type
        self.one_hot = one_hot
        self.action_bins = action_bins

        if self.type == 'gym':
            self.env = gym.make(name)
            # override environment default for max steps in an episode
            if max_steps:
                self.env._max_episode_steps = max_steps
        elif self.type == 'unity':
            # need to manually set seed to ensure a random environment is initialized
            self.env = UnityEnvironment(file_name=name, seed=random.randint(0, 2 ** 30))
            self.brain_name = self.env.brain_names[0]


    def reset(self):
        """Reset the environment."""

        if self.type == 'gym':
            state = self.env.reset()
            # one-hot encode state space (needed in environments where each state is numbered)
            if self.one_hot:
                state = np.eye(self.one_hot)[state]
        elif self.type == 'unity':
            info = self.env.reset(train_mode=True)[self.brain_name]
            if info.vector_observations:
                state = info.vector_observations[0]
            elif info.visual_observations:
                state = info.visual_observations[0]
                state = rgb2gray(state)
        return state

    def step(self, action):
        """Take a step in the environment.  Given an action, return the next state."""

        if self.type == 'gym':
            # convert discrete output from neural network to continuous action space
            if self.action_bins:
                action_grid = create_uniform_grid(self.env.action_space.low, self.env.action_space.high, bins=self.action_bins)
                state, reward, done, _ = self.env.step([action_grid[0][action]])
            else:
                state, reward, done, _ = self.env.step(action)
            # one-hot encode state space (needed in environments where each state is numbered)
            if self.one_hot:
                state = np.eye(self.one_hot)[state]
        elif self.type == 'unity':
            info = self.env.step(action)[self.brain_name]   # send the action to the environment
            if info.vector_observations:                    # get next state
                state = info.vector_observations[0]
            elif info.visual_observations:
                state = info.visual_observations[0]
                state = rgb2gray(state)
            reward = info.rewards[0]                        # get the reward
            done = info.local_done[0]
        return state, reward, done

    def render(self):
        """ Render the environment to visualize the agent interacting."""

        if self.type == 'gym':
            self.env.render()
        elif self.type == 'unity':
            pass
