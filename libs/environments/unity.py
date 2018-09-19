"""
UnityML Environments
"""

import time
import numpy as np
from unityagents import UnityEnvironment
from skimage.color import rgb2gray
import cv2
from libs.discretize import create_uniform_grid
from libs.visualize import show_frames_2d, show_frames_3d, show_frame

class UnityMLVector():
    """Define a UnityML vector based environment."""

    def __init__(self, name, seed):
        """ Initialize environment
        Params
        ======
            name (str): Environment name
            seed (int): Random seed
        """
        self.seed = seed
        print('SEED: {}'.format(self.seed))
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


class UnityMLVisual():
    """Define a UnityML visual (pixels based) environment.
       This is a simplified version of the one in environments_experimental.
       Frames from the environment are reshaped to fit torch Conv3D:
       (m, 1, 84, 84, 3) -> (m, 3, 1, 84, 84).
       Then 4 frames are stacked leading to a final state of shape:
       (m, 3, 4, 84, 84)."""

    def __init__(self, name, seed):
        """ Initialize environment
        Params
        ======
            name (str): Environment name
        """

        self.seed = seed
        print('SEED: {}'.format(self.seed))
        self.env = UnityEnvironment(file_name=name, seed=seed)
        self.brain_name = self.env.brain_names[0]
        self.full_state = np.zeros((1, 3, 4, 84, 84), dtype=np.uint8)


    def _add_frame(self, frame):
        """ Add a frame to a state.  Used for processing multiple states over time."""

        self.full_state[:, :, 3, :, : :] = self.full_state[:, :, 2, :, : :]
        self.full_state[:, :, 2, :, : :] = self.full_state[:, :, 1, :, : :]
        self.full_state[:, :, 1, :, : :] = self.full_state[:, :, 0, :, : :]
        self.full_state[:, :, 0, :, : :] = frame
        #return self.full_state

    def reset(self):
        """Reset the environment."""

        info = self.env.reset(train_mode=True)[self.brain_name]
        frame = info.visual_observations[0]
        frame = (frame * 255).astype(np.uint8)
        #print('reset frame before reshape:  {}'.format(frame.shape))
        frame = frame.reshape(1, 3, 84, 84)
        #print('reset frame afer reshape:  {}'.format(frame.shape))
        self._add_frame(frame)
        self._add_frame(frame)
        self._add_frame(frame)
        self._add_frame(frame)
        #print('reset:  {}'.format(self.full_state.shape))
        return self.full_state.copy()


    def step(self, action):
        """Take a step in the environment.  Given an action, return the next state."""

        info = self.env.step(action)[self.brain_name]   # send the action to the environment
        frame = info.visual_observations[0]
        frame = (frame * 255).astype(np.uint8)
        #print('step frame before reshape:  {}'.format(frame.shape))
        frame = frame.reshape(1, 3, 84, 84)
        #print('step frame afer reshape:  {}'.format(frame.shape))
        self._add_frame(frame)
        reward = info.rewards[0]                        # get the reward
        done = info.local_done[0]
        #print('step:  {}'.format(self.full_state.shape))
        return self.full_state.copy(), reward, done

    def render(self):
        """ Render the environment to visualize the agent interacting."""

        pass
