import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from libs.visualize import sub_plot


class Stats():
    """Base class for statistics.  Outputs to console and generates graphs."""

    def __init__(self):
        self.score = None
        self.avg_score = None
        self.std_dev = None
        self.scores = []                         # list containing scores from each episode
        self.avg_scores = []                     # list containing average scores after each episode
        self.scores_window = deque(maxlen=100)   # last 100 scores
        self.best_avg_score = -np.Inf            # best score for a single episode
        self.time_start = time.time()            # track cumulative wall time
        self.total_steps = 0                     # track cumulative steps taken

    def update(self, steps, rewards, i_episode):
        self.total_steps += steps
        self.score = sum(rewards)
        self.scores_window.append(self.score)
        self.scores.append(self.score)
        self.avg_score = np.mean(self.scores_window)
        self.avg_scores.append(self.avg_score)
        self.std_dev = np.std(self.scores_window)
        # update best average score
        if self.avg_score > self.best_avg_score and i_episode > 100:
            self.best_avg_score = self.avg_score

    def is_solved(self, i_episode, solve_score):
        return self.avg_score >= solve_score and i_episode >= 100

    def print_episode(self, i_episode, steps, stats_format, *args):
        common_stats = 'Episode: {:5}   Avg: {:8.2f}   BestAvg: {:8.2f}   σ: {:8.2f}  |  Steps: {:8}   Reward: {:8.2f}  |  '.format(i_episode, self.avg_score, self.best_avg_score, self.std_dev, steps, self.score)
        print('\r', common_stats, stats_format.format(*args), end="")

    def print_epoch(self, i_episode, stats_format, *args):
        n_secs = int(time.time() - self.time_start)
        common_stats = 'Episode: {:5}   Avg: {:8.2f}   BestAvg: {:8.2f}   σ: {:8.2f}  |  Steps: {:8}   Secs: {:6}      |  '.format(i_episode, self.avg_score, self.best_avg_score, self.std_dev, self.total_steps, n_secs)
        print('\r', common_stats, stats_format.format(*args))

    def print_solve(self, i_episode, stats_format, *args):
        self.print_epoch(i_episode, stats_format, *args)
        print('\nSolved in {:d} episodes!'.format(i_episode-100))

    def plot(self):
        plt.figure(1)
        sub_plot(211, self.scores, y_label='Score')
        sub_plot(212, self.avg_scores, y_label='Avg Score', x_label='Episodes')
        plt.show()


class DQNStats(Stats):
    """DQN specific graphs."""

    def plot(self, loss_list, entropy_list):
        window_size = len(loss_list) // 100 # window size is 1% of total steps
        plt.figure(1)
        # plot score
        sub_plot(231, self.scores, y_label='Score')
        sub_plot(234, self.avg_scores, y_label='Avg Score', x_label='Episodes')
        # plot loss
        sub_plot(232, loss_list, y_label='Loss')
        avg_loss = np.convolve(loss_list, np.ones((window_size,))/window_size, mode='valid')
        sub_plot(235, avg_loss, y_label='Avg Loss', x_label='Steps')
        # plot entropy
        sub_plot(233, entropy_list, y_label='Entropy')
        avg_entropy = np.convolve(entropy_list, np.ones((window_size,))/window_size, mode='valid')
        sub_plot(236, avg_entropy, y_label='Avg Entropy', x_label='Steps')
        plt.show()


class DDPGStats(Stats):
    """DDPG specific graphs."""

    def plot(self, loss_list):
        window_size = len(loss_list) // 100 # window size is 1% of total steps
        plt.figure(1)
        # plot score
        sub_plot(221, self.scores, y_label='Score')
        sub_plot(223, self.avg_scores, y_label='Avg Score', x_label='Episodes')
        # plot loss
        sub_plot(222, loss_list, y_label='Loss')
        avg_loss = np.convolve(loss_list, np.ones((window_size,))/window_size, mode='valid')
        sub_plot(224, avg_loss, y_label='Avg Loss', x_label='Steps')
        plt.show()


class MultiAgentDDPGStats(DDPGStats):
    """Provides optional debugging for multi agent DDPG."""

    def print_episode(self, i_episode, steps, stats_format, alpha, buffer_len, per_agent_rewards):
        Stats.print_episode(self, i_episode, steps, stats_format, alpha, buffer_len, per_agent_rewards)
        # DEBUG rewards for each agent
        #print('')
        #print(' '.join('%5.2f' % agent for agent in per_agent_rewards))
