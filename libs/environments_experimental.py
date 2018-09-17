"""
Classes to model various RL environments.  These models are considered more expiremental.
"""

import numpy as np
from unityagents import UnityEnvironment
from skimage.color import rgb2gray
from skimage.filters import gaussian
import gym
import gym.spaces
from discretize import create_uniform_grid
import cv2


class UnityMLVisualEnvironment():
    """Define a UnityML visual (pixels based) environment."""

    def __init__(self, name, seed):
        """ Initialize environment
        Params
        ======
            name (str): Environment name
        """

        self.env = UnityEnvironment(file_name=name, seed=seed)
        self.brain_name = self.env.brain_names[0]
        self.full_state = np.zeros((1, 1))

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
        #state = state.squeeze(0)                                              # (84, 84, 3)
        #state = cv2.resize(state, (42, 42), interpolation=cv2.INTER_AREA)     # (42, 42, 3)
        #state = np.expand_dims(state, axis=0)                                # (1, 42, 42, 3)

        # gaussian blur:                                        shape after
        #state = state.squeeze(0)                                # (84, 84, 3)
        #state = gaussian(state, sigma=0.75, multichannel=True)  # (84, 84, 3)
        #state = np.expand_dims(state, axis=0)                   # (1, 84, 84, 3)

        # crop and drop to single color channel
        #state = state.squeeze(0)                                 # (84, 84, 3)
        #state = state[42:85,  21:63, 0]                          # (42, 42)
        #state = np.expand_dims(state, axis=0)                    # (1, 42, 42)
        #state = np.expand_dims(state, axis=3)                    # (1, 42, 42, 1)

        # drop and highlight
        #state = state.squeeze(0)                                 # (84, 84, 3)
        #state = state[:, :, 0]                                   # (84, 84)
        #state[state >= 0.95] = 1.0                               # (84, 84)
        #state[state < 0.25] = 0.0                                # (84, 84)
        #state = np.expand_dims(state, axis=0)                    # (1, 84, 84)
        #state = np.expand_dims(state, axis=3)                    # (1, 84, 84, 1)

        # crop, resize, gaussian
        #state = state.squeeze(0)
        #state = state[42:85, :, :]
        #state = cv2.resize(state, (42, 42), interpolation = cv2.INTER_AREA)
        #state = gaussian(state, multichannel=True)
        #state = np.expand_dims(state, axis=0)

        return state


    def _get_state(self, info):
        state = info.visual_observations[0]
        state = self._preprocess(state)
        return state

    def _add_frame(self, frame):
        """ Add a frame to a state.  Used for processing multiple states over time."""

        self.full_state[:, 0, :, :, : :] = self.full_state[:, 1, :, :, : :]
        self.full_state[:, 1, :, :, : :] = self.full_state[:, 2, :, :, : :]
        self.full_state[:, 2, :, :, : :] = self.full_state[:, 3, :, :, : :]
        self.full_state[:, 3, :, :, : :] = frame
        return self.full_state

    def reset(self):
        """Reset the environment."""

        info = self.env.reset(train_mode=True)[self.brain_name]

        #################
        # process 1 frame
        #state = self._get_state(info)
        #return state

        # process 4 frames
        frame = self._get_state(info)
        self.full_state = np.stack((frame, frame, frame, frame), axis=1)
        return self.full_state
        #################

    def step(self, action):
        """Take a step in the environment.  Given an action, return the next state."""

        info = self.env.step(action)[self.brain_name]   # send the action to the environment

        #################
        # process 1 frame
        #state = self._get_state(info)

        # process 4 frames
        frame = self._get_state(info)
        state = self._add_frame(frame)
        #################

        reward = info.rewards[0]                        # get the reward
        done = info.local_done[0]
        return state, reward, done

    def render(self):
        """ Render the environment to visualize the agent interacting."""

        pass
