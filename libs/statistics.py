import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from libs.visualize import sub_plot
from tensorboardX import SummaryWriter


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
        self.writer = SummaryWriter()
        self.log_file_name = 'output.txt'
        # zero out current output file
        f = open(self.log_file_name,'w')
        f.close()

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
        common_stats = 'Episode: {:5}   Avg: {:8.3f}   BestAvg: {:8.3f}   σ: {:8.3f}  |  Steps: {:8}   Reward: {:8.3f}  |  '.format(i_episode, self.avg_score, self.best_avg_score, self.std_dev, steps, self.score)
        print( '\r' + common_stats + stats_format.format(*args), end="")
        self.writer.add_scalar('data/reward', self.score, i_episode)
        self.writer.add_scalar('data/std_dev', self.std_dev, i_episode)
        self.writer.add_scalar('data/avg_reward', self.avg_score, i_episode)
        print('')  # print every episode

    def print_epoch(self, i_episode, stats_format, *args):
        n_secs = int(time.time() - self.time_start)
        common_stats = 'Episode: {:5}   Avg: {:8.3f}   BestAvg: {:8.3f}   σ: {:8.3f}  |  Steps: {:8}   Secs: {:6}      |  '.format(i_episode, self.avg_score, self.best_avg_score, self.std_dev, self.total_steps, n_secs)
        print('\r' + common_stats + stats_format.format(*args))
        print(common_stats, stats_format.format(*args), file=open(self.log_file_name,'a'))

    def print_solve(self, i_episode, stats_format, *args):
        self.print_epoch(i_episode, stats_format, *args)
        print('\nSolved in {:d} episodes!'.format(i_episode-100))
        print('\nSolved in {:d} episodes!'.format(i_episode-100), file=open(self.log_file_name,'a'))

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

class PPOStats(Stats):
    """DDPG specific graphs."""

    def print_episode(self, i_episode, steps, stats_format, *args):
        Stats.print_episode(self, i_episode, steps, stats_format, *args)
        epsilon, beta = args
        self.writer.add_scalar('data/epsilon', epsilon, i_episode)
        self.writer.add_scalar('data/beta', beta, i_episode)


class DDPGStats(Stats):
    """DDPG specific graphs."""

    def print_episode(self, i_episode, steps, stats_format, *args):
        Stats.print_episode(self, i_episode, steps, stats_format, *args)
        alpha, buffer_len = args
        self.writer.add_scalar('data/alpha', alpha, i_episode)
        self.writer.add_scalar('data/buffer_len', buffer_len, i_episode)

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


class MultiAgentDDPGv1Stats(DDPGStats):
    """Provides optional debugging for multi agent DDPG."""

    def print_episode(self, i_episode, steps, stats_format, *args):
        Stats.print_episode(self, i_episode, steps, stats_format, *args)
        alpha, buffer_len, per_agent_rewards = args
        self.writer.add_scalar('data/alpha', alpha, i_episode)
        self.writer.add_scalar('data/buffer_len', buffer_len, i_episode)
        # DEBUG rewards for each agent
        #print('')
        #print(' '.join('%5.2f' % agent for agent in per_agent_rewards))


class MultiAgentDDPGv2Stats(DDPGStats):
    """Adds additional logging via tensorboard."""

    def print_episode(self, i_episode, steps, stats_format, *args):
        buffer_len, noise_weight, critic_loss_01, critic_loss_02, actor_loss_01, actor_loss_02, noise_val_01, noise_val_02, rewards_01, rewards_02 = args
        Stats.print_episode(self, i_episode, steps, stats_format, buffer_len, noise_weight)
        #alpha, buffer_len, per_agent_rewards = args
        # log lots of stuff to tensorboard
        self.writer.add_scalar('global/reward', self.score, i_episode)
        self.writer.add_scalar('global/std_dev', self.std_dev, i_episode)
        self.writer.add_scalar('global/avg_reward', self.avg_score, i_episode)
        self.writer.add_scalar('global/buffer_len', buffer_len, i_episode)
        self.writer.add_scalar('global/noise_weight', noise_weight, i_episode)
        self.writer.add_scalar('agent_01/critic_loss', critic_loss_01, i_episode)
        self.writer.add_scalar('agent_02/critic_loss', critic_loss_02, i_episode)
        self.writer.add_scalar('agent_01/actor_loss', actor_loss_01, i_episode)
        self.writer.add_scalar('agent_02/actor_loss', actor_loss_02, i_episode)
        self.writer.add_scalar('agent_01/noise_val_01', noise_val_01[0], i_episode)
        self.writer.add_scalar('agent_01/noise_val_02', noise_val_01[1], i_episode)
        self.writer.add_scalar('agent_02/noise_val_01', noise_val_02[0], i_episode)
        self.writer.add_scalar('agent_02/noise_val_02', noise_val_02[1], i_episode)
        self.writer.add_scalar('agent_01/reward', rewards_01, i_episode)
        self.writer.add_scalar('agent_02/reward', rewards_02, i_episode)
        # DEBUG rewards for each agent
        #print('')
        #print(' '.join('%5.2f' % agent for agent in per_agent_rewards))
