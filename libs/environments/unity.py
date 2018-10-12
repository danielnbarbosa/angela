"""
UnityML Environments
"""

import numpy as np
from unityagents import UnityEnvironment
from libs.visualize import show_frames_2d, show_frames_3d, show_frame

class UnityML():
    """Base class for Unity ML environments."""

    def __init__(self, name, seed=0):
        self.seed = seed
        print('SEED: {}'.format(self.seed))
        self.env = UnityEnvironment(file_name=name, seed=seed)
        self.brain_name = self.env.brain_names[0]

    def render(self):
        """
        Render the environment to visualize the agent interacting.
        Does nothing because rendering is always on as is required by linux environments.
        """
        pass


class UnityMLVector(UnityML):
    """
    UnityML environment with vector observations.
    state is 1-D numpy array.  reward and done are scalars.
    """

    def reset(self):
        """Reset the environment."""
        info = self.env.reset(train_mode=True)[self.brain_name]
        state = info.vector_observations[0]
        return state

    def step(self, action):
        """Take a step in the environment."""
        info = self.env.step(action)[self.brain_name]
        state = info.vector_observations[0]
        reward = info.rewards[0]
        done = info.local_done[0]
        return state, reward, done


class UnityMLVectorMultiAgent(UnityML):
    """
    Multi-agent UnityML environment with vector observations.
    state is 2-D numpy array.  reward and done are lists.
    """

    def reset(self):
        """Reset the environment."""
        info = self.env.reset(train_mode=True)[self.brain_name]
        state = info.vector_observations
        return state

    def step(self, action):
        """Take a step in the environment."""
        info = self.env.step(action)[self.brain_name]
        state = info.vector_observations
        reward = info.rewards
        done = info.local_done
        return state, reward, done


class UnityMLVisual(UnityML):
    """
    UnityML environment with visual observations.
    Takes 84x84 RGB frames output from visual observations and stacks 4 into a state.
    Final state is shaped to fit torch Conv3D: (m, 3, 4, 84, 84)
    """

    def __init__(self, name, seed=0):
        super(UnityMLVisual, self).__init__(name, seed)
        self.full_state = np.zeros((1, 3, 4, 84, 84), dtype=np.uint8)

    def _add_frame(self, frame):
        """ Add a frame to a state.  Used for processing multiple states over time."""
        self.full_state[:, :, 3, :, : :] = self.full_state[:, :, 2, :, : :]
        self.full_state[:, :, 2, :, : :] = self.full_state[:, :, 1, :, : :]
        self.full_state[:, :, 1, :, : :] = self.full_state[:, :, 0, :, : :]
        self.full_state[:, :, 0, :, : :] = frame

    def reset(self):
        """Reset the environment."""
        info = self.env.reset(train_mode=True)[self.brain_name]
        frame = info.visual_observations[0]
        frame = (frame * 255).astype(np.uint8)
        #print('reset frame before reshape:  {}'.format(frame.shape))  # DEBUG
        frame = frame.reshape(1, 3, 84, 84)
        #print('reset frame afer reshape:  {}'.format(frame.shape))  # DEBUG
        self._add_frame(frame)
        self._add_frame(frame)
        self._add_frame(frame)
        self._add_frame(frame)
        #print('reset:  {}'.format(self.full_state.shape))  # DEBUG
        return self.full_state.copy()

    def step(self, action):
        """Take a step in the environment."""
        info = self.env.step(action)[self.brain_name]
        frame = info.visual_observations[0]
        frame = (frame * 255).astype(np.uint8)  # save space by storing frames as uint8
        #print('step frame before reshape:  {}'.format(frame.shape))  # DEBUG
        frame = frame.reshape(1, 3, 84, 84)
        #print('step frame afer reshape:  {}'.format(frame.shape))  # DEBUG
        self._add_frame(frame)
        reward = info.rewards[0]
        done = info.local_done[0]
        #print('step:  {}'.format(self.full_state.shape))  # DEBUG
        return self.full_state.copy(), reward, done
