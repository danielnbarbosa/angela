import time
import numpy as np
from collections import deque

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
        # update stats
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


    ############ DQN style methods ############
    def print_dqn_episode(self, i_episode, epsilon, alpha, buffer_len):
        print('\rEpisode {:5}   Avg: {:7.2f}   BestAvg: {:7.2f}   σ: {:7.2f}'
              '   |   ε: {:6.4f}  ⍺: {:6.4f}  Buffer: {:6}   Now: {:7.2f}'
              .format(i_episode, self.avg_score, self.best_avg_score, self.std_dev, epsilon, alpha, buffer_len, self.score), end="")


    def print_dqn_epoch(self, i_episode, epsilon, alpha, buffer_len):
        n_secs = int(time.time() - self.time_start)
        print('\rEpisode {:5}   Avg: {:7.2f}   BestAvg: {:7.2f}   σ: {:7.2f}'
              '   |   ε: {:6.4f}  ⍺: {:6.4f}  Buffer: {:6}   Steps: {:8}  Secs: {:6}'
              .format(i_episode, self.avg_score, self.best_avg_score, self.std_dev, epsilon, alpha, buffer_len, self.total_steps, n_secs))


    def print_dqn_solve(self, i_episode, epsilon, alpha, buffer_len):
        self.print_dqn_epoch(i_episode, epsilon, alpha, buffer_len)
        print('\nSolved in {:d} episodes!'.format(i_episode-100))


    ############ HC style methods ############
    def print_hc_episode(self, i_episode, pop_best_return, best_return, noise_scale):
        print('\rEpisode {:5}   Avg: {:7.2f}   BestAvg: {:7.2f}   σ: {:7.2f}'
              '   |   Best: {:7.2}   Noise: {:6.4f}   Now: {:7.2f}'
              .format(i_episode, self.avg_score, self.best_avg_score, self.std_dev, best_return, noise_scale, pop_best_return), end="")


    def print_hc_epoch(self, i_episode, pop_best_return, best_return, noise_scale):
        n_secs = int(time.time() - self.time_start)
        print('\rEpisode {:5}   Avg: {:7.2f}   BestAvg: {:7.2f}   σ: {:7.2f}'
              '   |   Best: {:7.2}   Noise: {:6.4f}   Steps: {:8}  Secs: {:6}'
              .format(i_episode, self.avg_score, self.best_avg_score, self.std_dev, best_return, noise_scale, self.total_steps, n_secs))


    def print_hc_solve(self, i_episode, pop_best_return, best_return, noise_scale):
        self.print_hc_epoch(i_episode, pop_best_return, best_return, noise_scale)
        print('\nSolved in {:d} episodes!'.format(i_episode-100))


    ############ PG style methods ############
    def print_pg_episode(self, i_episode):
        print('\rEpisode {:5}   Avg: {:7.2f}   BestAvg: {:7.2f}   σ: {:7.2f}'
              '   |   Now: {:7.2f}'
              .format(i_episode, self.avg_score, self.best_avg_score, self.std_dev, self.score), end="")


    def print_pg_epoch(self, i_episode):
        n_secs = int(time.time() - self.time_start)
        print('\rEpisode {:5}   Avg: {:7.2f}   BestAvg: {:7.2f}   σ: {:7.2f}'
              '   |   Steps: {:8}  Secs: {:6}'
              .format(i_episode, self.avg_score, self.best_avg_score, self.std_dev, self.total_steps, n_secs))


    def print_pg_solve(self, i_episode):
        self.print_pg_epoch(i_episode)
        print('\nSolved in {:d} episodes!'.format(i_episode-100))

    ### Generic methods ###
    def is_solved(self, i_episode, solve_score):
        return self.avg_score >= solve_score and i_episode >= 100
