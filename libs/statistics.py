import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from libs.visualize import sub_plot


class Stats():

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



class DeepQNetworkStats(Stats):
    def print_episode(self, i_episode, epsilon, alpha, buffer_len, steps):
        print('\rEpisode {:5}   Avg: {:8.2f}   BestAvg: {:8.2f}   σ: {:8.2f}'
              '   |   ε: {:6.4f}  ⍺: {:6.4f}  Buffer: {:6}   Reward: {:8.2f}   Steps: {:6}'
              .format(i_episode, self.avg_score, self.best_avg_score, self.std_dev,
                      epsilon, alpha, buffer_len, self.score, steps), end="")

    def print_epoch(self, i_episode, epsilon, alpha, buffer_len):
        n_secs = int(time.time() - self.time_start)
        print('\rEpisode {:5}   Avg: {:8.2f}   BestAvg: {:8.2f}   σ: {:8.2f}'
              '   |   ε: {:6.4f}  ⍺: {:6.4f}  Buffer: {:6}'
              '   |   Steps: {:8}  Secs: {:6}'
              .format(i_episode, self.avg_score, self.best_avg_score, self.std_dev,
                      epsilon, alpha, buffer_len,
                      self.total_steps, n_secs))

    def print_solve(self, i_episode, epsilon, alpha, buffer_len):
        self.print_epoch(i_episode, epsilon, alpha, buffer_len)
        print('\nSolved in {:d} episodes!'.format(i_episode-100))

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


class HillClimbingStats(Stats):
    def print_episode(self, i_episode, pop_best_return, max_best_return, noise_scale, steps):
        print('\rEpisode {:5}   Avg: {:8.2f}   BestAvg: {:8.2f}   σ: {:8.2f}'
              '   |   Best: {:8.2f}   Noise: {:6.4f}   Reward: {:8.2f}   Steps: {:6}'
              .format(i_episode, self.avg_score, self.best_avg_score, self.std_dev,
                      max_best_return, noise_scale, pop_best_return, steps), end="")

    def print_epoch(self, i_episode, max_best_return, noise_scale):
        n_secs = int(time.time() - self.time_start)
        print('\rEpisode {:5}   Avg: {:8.2f}   BestAvg: {:8.2f}   σ: {:8.2f}'
              '   |   Best: {:8.2f}   Noise: {:6.4f}'
              '   |   Steps: {:8}  Secs: {:6}'
              .format(i_episode, self.avg_score, self.best_avg_score, self.std_dev,
                      max_best_return, noise_scale,
                      self.total_steps, n_secs))

    def print_solve(self, i_episode, max_best_return, noise_scale):
        self.print_epoch(i_episode, max_best_return, noise_scale)
        print('\nSolved in {:d} episodes!'.format(i_episode-100))

    def plot(self):
        plt.figure(1)
        sub_plot(211, self.scores, y_label='Score')
        sub_plot(212, self.avg_scores, y_label='Avg Score', x_label='Episodes')
        plt.show()


class PolicyGradientStats(Stats):
    def print_episode(self, i_episode, steps):
        print('\rEpisode {:5}   Avg: {:8.2f}   BestAvg: {:8.2f}   σ: {:8.2f}'
              '   |   Reward: {:8.2f}   Steps: {:6}'
              .format(i_episode, self.avg_score, self.best_avg_score,
                      self.std_dev, self.score, steps), end="")

    def print_epoch(self, i_episode):
        n_secs = int(time.time() - self.time_start)
        print('\rEpisode {:5}   Avg: {:8.2f}   BestAvg: {:8.2f}   σ: {:8.2f}'
              '   |   Steps: {:8}  Secs: {:6}     '
              .format(i_episode, self.avg_score, self.best_avg_score,
                      self.std_dev,
                      self.total_steps, n_secs))

    def print_solve(self, i_episode):
        self.print_epoch(i_episode)
        print('\nSolved in {:d} episodes!'.format(i_episode-100))

    def plot(self):
        plt.figure(1)
        sub_plot(211, self.scores, y_label='Score')
        sub_plot(212, self.avg_scores, y_label='Avg Score', x_label='Episodes')
        plt.show()
