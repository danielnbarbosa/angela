"""
Functions to execute training and visualize agent.
"""

import time
from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt
#import simpleaudio as sa



def train(environment, agent, n_episodes=2000, max_t=1000,
          eps_start=1.0,
          eps_end=0.01,
          eps_decay=0.995,
          render_every=100000,
          solve_score=100000.0,
          sound_when_done=False,
          graph_when_done=False):
    """ Run training loop.

    Params
    ======
        environment: environment object
        agent: agent object
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        render_every (int): render the agent interacting in the environment every n episodes
        solve_score (float): criteria for considering the environment solved
        sound_when_done (bool): wheter to play a sound to announce training is finished
        graph_when_done (bool): whether to show matplotlib graphs of the training run
    """
    scores = []                         # list containing scores from each episode
    avg_scores = []                     # list containing average scores after each episode
    scores_window = deque(maxlen=100)   # last 100 scores
    eps = eps_start                     # initialize epsilon
    best_avg_score = -10000             # best score for a single episode
    time_start = time.time()            # track wall time over 100 episodes
    total_steps = 0                     # track steps taken over 100 episodes

    for i_episode in range(1, n_episodes+1):
        # reset environment
        state = environment.reset()
        score = 0
        # loop over steps
        for t in range(max_t):
            # render during training
            if i_episode % render_every == 0:
                environment.render()

            # visualize each frame
            #print(state.shape)
            #plt.figure(1)
            #sub_plot_img(221, state.squeeze(0)[0], x_label='0')
            #sub_plot_img(222, state.squeeze(0)[1], x_label='1')
            #sub_plot_img(223, state.squeeze(0)[2], x_label='2')
            #sub_plot_img(224, state.squeeze(0)[3], x_label='3')
            #plt.show()

            # select an action
            action = agent.act(state, eps)
            # take action in environment
            next_state, reward, done = environment.step(action)
            # update agent with returned information
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                total_steps += t
                break

        # decrease epsilon
        eps = max(eps_end, eps_decay*eps)

        # update stats
        scores_window.append(score)
        scores.append(score)
        avg_score = np.mean(scores_window)
        avg_scores.append(avg_score)
        buffer_len = len(agent.memory)
        # update best average score, let a few episodes pass in case you get lucky early
        if avg_score > best_avg_score and i_episode > 30:
            best_avg_score = avg_score

        # print stats every episode
        print('\rEpisode {:5}\tAvg: {:5.2f}\tBest: {:5.2f}'
              '\tε: {:.4f}  ⍺: {:.4f}  Buffer: {:6}'
              .format(i_episode, avg_score, best_avg_score, eps, agent.alpha, buffer_len), end="")

        # every 100 episodes
        if i_episode % 100 == 0:
            # calculate wall time
            n_secs = int(time.time() - time_start)
            # print extented stats
            print('\rEpisode {:5}\tAvg: {:5.2f}\tBest: {:5.2f}'
                  '\tε: {:.4f}  ⍺: {:.4f}  Buffer: {:6}  Steps: {:6}  Secs: {:4}'
                  .format(i_episode, avg_score, best_avg_score, eps, agent.alpha, buffer_len, total_steps, n_secs))
            save_name = '../checkpoints/episode.{}.pth'.format(i_episode)
            torch.save(agent.qnetwork_local.state_dict(), save_name)
            # reset counters
            time_start = time.time()
            total_steps = 0

        # if solved
        if avg_score >= solve_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tStdDev: {:.2f}'
                  .format(i_episode-100, avg_score, np.std(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), '../checkpoints/solved.pth')
            break

    # play sound to signal training is finished
    if sound_when_done:
        play_sound('../libs/fanfare.wav')
    if graph_when_done:
        plot(scores, avg_scores, agent.loss_list, agent.entropy_list)


def play_sound(file):
    """ Play a sound. """

    pass
    #wave_obj = sa.WaveObject.from_wave_file(file)
    #play_obj = wave_obj.play()
    #play_obj.wait_done()

def sub_plot_img(coords, img, y_label='', x_label=''):
    """ Plot a single image (subplot). """

    plt.subplot(coords)
    plt.imshow(img)
    plt.ylabel(y_label)
    plt.xlabel(x_label)

def sub_plot(coords, data, y_label='', x_label=''):
    """ Plot a single graph (subplot). """

    plt.subplot(coords)
    plt.plot(np.arange(len(data)), data)
    plt.ylabel(y_label)
    plt.xlabel(x_label)

def plot(scores, avg_scores, loss_list, entropy_list):
    """ Plot all data from training run. """

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


def watch(environment, agent, checkpoints, frame_sleep=0.05):
    """ Visualize agent using saved checkpoints. """

    for checkpoint in checkpoints:
        # load saved weights from file
        print('Watching: {}'.format(checkpoint))
        agent.qnetwork_local.load_state_dict(torch.load('../checkpoints/' + checkpoint + '.pth'))
        # reset environment
        state = environment.reset()
        # interact with environment
        for _ in range(600):
            # select an action
            action = agent.act(state)
            time.sleep(frame_sleep)
            # take action in environment
            state, _, done = environment.step(action)
            environment.render()
            if done:
                break

def load(model, file_name):
    """ Load saved model weights from a checkpoint file """

    model.local.load_state_dict(torch.load('../checkpoints/' + file_name))
    model.target.load_state_dict(torch.load('../checkpoints/' + file_name))
