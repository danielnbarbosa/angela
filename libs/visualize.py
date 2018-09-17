"""
Utility functions for generating training graphs and peeking at agent internals.
"""

import numpy as np
import matplotlib.pyplot as plt

def sub_plot(coords, data, y_label='', x_label=''):
    """Plot a single graph (subplot)."""

    plt.subplot(coords)
    plt.plot(np.arange(len(data)), data)
    plt.ylabel(y_label)
    plt.xlabel(x_label)


def sub_plot_img(coords, img, y_label='', x_label=''):
    """Plot a single image (subplot)."""

    plt.subplot(coords)
    plt.imshow(img)
    plt.ylabel(y_label)
    plt.xlabel(x_label)


def show_frames(state):
    """Show each of the frames being passed as a single state."""
    plt.figure(1)
    sub_plot_img(141, state[0, :, 0, :, : :].reshape(84, 84, 3), x_label='0')
    sub_plot_img(142, state[0, :, 1, :, : :].reshape(84, 84, 3), x_label='1')
    sub_plot_img(143, state[0, :, 2, :, : :].reshape(84, 84, 3), x_label='2')
    sub_plot_img(144, state[0, :, 3, :, : :].reshape(84, 84, 3), x_label='3')
    plt.show()


def show_frames_pg(now, state):
    """Show each of the frames being passed as a single state."""
    plt.figure(1)
    sub_plot_img(131, now[0, :, ].reshape(80, 80), x_label='now')
    sub_plot_img(132, state[0, 0, :, ].reshape(80, 80), x_label='0')
    sub_plot_img(133, state[0, 1, :, ].reshape(80, 80), x_label='1')
    plt.show()
