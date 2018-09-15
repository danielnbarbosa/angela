"""
Utility functions for generating training graphs and peeking at agent internals.
"""

import numpy as np
import matplotlib.pyplot as plt
#import simpleaudio as sa


def sub_plot_img(coords, img, y_label='', x_label=''):
    """Plot a single image (subplot)."""

    plt.subplot(coords)
    plt.imshow(img)
    plt.ylabel(y_label)
    plt.xlabel(x_label)


def sub_plot(coords, data, y_label='', x_label=''):
    """Plot a single graph (subplot)."""

    plt.subplot(coords)
    plt.plot(np.arange(len(data)), data)
    plt.ylabel(y_label)
    plt.xlabel(x_label)


def plot_dqn(scores, avg_scores, loss_list, entropy_list):
    """Plot all data from training run."""

    window_size = len(loss_list) // 100 # window size is 1% of total steps
    plt.figure(1)
    # plot score
    sub_plot(231, scores, y_label='Score')
    sub_plot(234, avg_scores, y_label='Avg Score', x_label='Episodes')
    # plot loss
    sub_plot(232, loss_list, y_label='Loss')
    avg_loss = np.convolve(loss_list, np.ones((window_size,))/window_size, mode='valid')
    sub_plot(235, avg_loss, y_label='Avg Loss', x_label='Steps')
    # plot entropy
    sub_plot(233, entropy_list, y_label='Entropy')
    avg_entropy = np.convolve(entropy_list, np.ones((window_size,))/window_size, mode='valid')
    sub_plot(236, avg_entropy, y_label='Avg Entropy', x_label='Steps')

    plt.show()


def plot_hc(scores, avg_scores):
    """Plot all data from training run."""

    plt.figure(1)
    # plot score
    sub_plot(211, scores, y_label='Score')
    sub_plot(212, avg_scores, y_label='Avg Score', x_label='Episodes')

    plt.show()


def plot_pg(scores, avg_scores):
    """Plot all data from training run."""

    plt.figure(1)
    # plot score
    sub_plot(211, scores, y_label='Score')
    sub_plot(212, avg_scores, y_label='Avg Score', x_label='Episodes')

    plt.show()


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


def play_sound(file):
    """ Play a sound. """

    wave_obj = sa.WaveObject.from_wave_file(file)
    play_obj = wave_obj.play()
    play_obj.wait_done()
