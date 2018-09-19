"""
PyGame Learning Environments
"""

import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
import cv2
from libs.visualize import show_frames_2d, show_frames_3d, show_frame


class PLEFlappyBird():
    """
    PyGame Learning Environment for use only with FlappyBird.
    Does pre-processing specific to FlappyBird game.
    """

    def __init__(self, seed, render=False):
        self.seed = seed
        print('SEED: {}'.format(self.seed))
        game = FlappyBird(pipe_gap=150)
        self.env = PLE(game, fps=30, display_screen=render)
        # TODO: figure out how to pass seed.  it's not using rng=seed in PLE()
        self.env.init()
        self.full_state = np.zeros((1, 4, 80, 80), dtype=np.uint8)
        self.frame_sleep = 0.02

    def _prepro(self, frame):
        """Pre-process 288x512x3 uint8 frame into 80x80 uint8 frame."""
        frame = frame[:, :, 2]   # drop to one color channel
        frame = frame.T          # rotate 90 degrees
        frame[frame == 140] = 0  # filter out background
        frame[frame == 147] = 0
        frame[frame == 160] = 0
        frame[frame == 194] = 0
        frame[frame == 210] = 0
        frame[frame != 0] = 255    # set everything else to 255
        frame = cv2.resize(frame, (80, 80))  # downsample
        #show_frame(frame)  # DEBUG
        return frame

    def _add_frame(self, frame):
        """ Add single frame to state.  Used for processing multiple states over time."""
        self.full_state[:, 3, :, : :] = self.full_state[:, 2, :, : :]
        self.full_state[:, 2, :, : :] = self.full_state[:, 1, :, : :]
        self.full_state[:, 1, :, : :] = self.full_state[:, 0, :, : :]
        self.full_state[:, 0, :, : :] = frame

    def reset(self):
        """Reset the environment."""
        self.env.reset_game()
        frame = self.env.getScreenRGB()
        #print('reset() frame from environment:  {}'.format(frame.shape))  # DEBUG
        frame = self._prepro(frame)
        #print('reset() frame after _prepro():  {}'.format(frame.shape))  # DEBUG
        frame = np.expand_dims(frame, axis=0)
        #print('reset() frame after reshape:  {}'.format(frame.shape))  # DEBUG
        self._add_frame(frame)
        self._add_frame(frame)
        self._add_frame(frame)
        self._add_frame(frame)
        #print('reset():  {}'.format(self.full_state.shape))  # DEBUG
        #show_frames_2d(self.full_state)  # DEBUG
        return self.full_state.copy()

    def step(self, action):
        """Take a step in the environment."""
        reward = self.env.act(action)
        frame = self.env.getScreenRGB()
        done = True if self.env.game_over() else False
        #print('step() frame from environment:  {}'.format(frame))  # DEBUG
        frame = self._prepro(frame)
        #print('step() frame after _prepro():  {}'.format(frame))  # DEBUG
        frame = np.expand_dims(frame, axis=0)
        #print('step() frame after reshape:  {}'.format(frame))  # DEBUG
        self._add_frame(frame)
        #print('step():  {}'.format(self.full_state))  # DEBUG
        #show_frames_2d(self.full_state)  # DEBUG
        return self.full_state.copy(), reward, done

    def render(self):
        """
        Render the environment to visualize the agent interacting.
        Does nothing because rendering is handled by setting display_screen=True
        when creating the PLE() object.
        """
        pass
