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

    def __init__(self, name, max_steps=None, one_hot=None, action_bins=None):
        """ Initialize environment
        Params
        ======
            name (str): Environment name
            max_steps (int): Maximum number of steps to run before returning done
            one_hot (int): Size of 1-D one-hot vector
            action_bins (tuple): Number of splits to divide each dimension of continuous space
        """
        self.one_hot = one_hot
        self.action_bins = action_bins
        self.env = gym.make(name)
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




class UnityMLEnvironment():
    """Define a UnityML environment."""

    def __init__(self, name, observations):
        """ Initialize environment
        Params
        ======
            name (str): Environment name
            observations (str): Observation type: vector, visual
        """

        # need to manually set seed to ensure a random environment is initialized
        seed = random.randint(0, 2 ** 30)
        self.env = UnityEnvironment(file_name=name, seed=seed)
        self.brain_name = self.env.brain_names[0]
        self.observations = observations

    def _preprocess(self, state):
        """
        Input dimensions from Unity visual environment are: (1, 84, 84, 3)
        Preprocessing always reshapes dimensions back to: (m, h, w, c)
        The model then transforms them to be processed by conv2d
        """

        # greyscale:                            shape after
        #state = state.squeeze(0)               # (84, 84, 3)
        #state = rgb2gray(state)                # (84, 84)
        #state = np.expand_dims(state, axis=0)  # (1, 84, 84)
        #state = np.expand_dims(state, axis=3)  # (1, 84, 84, 1)

        # downsize:                                                           shape after
        #state = state.squeeze(0)                                             # (84, 84, 3)
        #state = cv2.resize(state, (42, 42), interpolation = cv2.INTER_AREA)  # (42, 42, 3)
        #state = np.expand_dims(state, axis=0)                                # (1, 42, 42, 3)

        # gaussian blur:                                        shape after
        state = state.squeeze(0)                                # (84, 84, 3)
        state = gaussian(state, sigma=0.75, multichannel=True)  # (84, 84, 3)
        state = np.expand_dims(state, axis=0)                   # (1, 84, 84, 3)

        return state


    def _get_state(self, info):
        if self.observations == 'vector':
            state = info.vector_observations[0]
        elif self.observations == 'visual':
            state = info.visual_observations[0]
            state = self._preprocess(state)
        return state

    def reset(self):
        """Reset the environment."""

        info = self.env.reset(train_mode=True)[self.brain_name]
        state = self._get_state(info)
        return state

    def step(self, action):
        """Take a step in the environment.  Given an action, return the next state."""

        info = self.env.step(action)[self.brain_name]   # send the action to the environment
        state = self._get_state(info)
        reward = info.rewards[0]                        # get the reward
        done = info.local_done[0]
        return state, reward, done

    def render(self):
        """ Render the environment to visualize the agent interacting."""

        pass
