"""
Classes to model various RL environments.
"""

import random
import numpy as np
from unityagents import UnityEnvironment
from skimage.color import rgb2gray
from skimage.filters import gaussian
import gym
from discretize import create_uniform_grid
import cv2


class GymEnvironment():
    """Define an OpenAI Gym environment."""

    def __init__(self, name, seed, max_steps=None, one_hot=None, action_bins=None):
        """ Initialize environment
        Params
        ======
            name (str): Environment name
            seed (int): Random seed
            max_steps (int): Maximum number of steps to run before returning done
            one_hot (int): Size of 1-D one-hot vector
            action_bins (tuple): Number of splits to divide each dimension of continuous space
        """
        self.one_hot = one_hot
        self.action_bins = action_bins

        self.env = gym.make(name)
        #self.env = gym.wrappers.Monitor(self.env, "recording")
        self.env.seed(seed)
        # override environment default for max steps in an episode
        if max_steps:
            self.env._max_episode_steps = max_steps


    def reset(self):
        """Reset the environment."""

        state = self.env.reset()
        # one-hot encode state space (needed in environments where each state is numbered)
        if self.one_hot:
            state = np.eye(self.one_hot)[state]
        return state


    def step(self, action):
        """Take a step in the environment.  Given an action, return the next state."""

        # convert discrete output from neural network to continuous action space
        if self.action_bins:
            action_grid = create_uniform_grid(self.env.action_space.low,
                                              self.env.action_space.high,
                                              bins=self.action_bins)
            state, reward, done, _ = self.env.step([action_grid[0][action]])
        else:
            state, reward, done, _ = self.env.step(action)
        # one-hot encode state space (needed in environments where each state is numbered)
        if self.one_hot:
            state = np.eye(self.one_hot)[state]
        return state, reward, done

    def render(self):
        """ Render the environment to visualize the agent interacting."""

        self.env.render()


class UnityMLVectorEnvironment():
    """Define a UnityML vector based environment."""

    def __init__(self, name, seed):
        """ Initialize environment
        Params
        ======
            name (str): Environment name
            seed (int): Random seed
        """

        self.env = UnityEnvironment(file_name=name, seed=seed)
        self.brain_name = self.env.brain_names[0]
        self.full_state = np.zeros((1, 1))

    def reset(self):
        """Reset the environment."""

        info = self.env.reset(train_mode=True)[self.brain_name]
        state = info.vector_observations[0]
        return state


    def step(self, action):
        """Take a step in the environment.  Given an action, return the next state."""

        info = self.env.step(action)[self.brain_name]   # send the action to the environment
        state = info.vector_observations[0]
        reward = info.rewards[0]                        # get the reward
        done = info.local_done[0]
        return state, reward, done


    def render(self):
        """ Render the environment to visualize the agent interacting."""

        pass
