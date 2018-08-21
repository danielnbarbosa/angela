import time
from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt
import simpleaudio as sa



def train(env, agent, env_type='gym', brain_name=None, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, render_every=1000000, solve_score=10000.0, graph_results=False):
    """ Run training loop.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        solve_score (float): criteria for considering the environment solved
    """
    scores = []                         # list containing scores from each episode
    avg_scores = []                     # list containing average scores after each episode
    scores_window = deque(maxlen=100)   # last 100 scores
    eps = eps_start                     # initialize epsilon
    best_avg_score = -100000            # best score for a single episode
    time_start = time.time()            # track wall time over 100 episodes
    total_steps = 0                     # track steps taken over 100 episodes

    for i_episode in range(1, n_episodes+1):
        state = initialize_env(env, env_type, brain_name)
        score = 0
        # loop over steps
        for t in range(max_t):
            # render during training
            if i_episode % render_every == 0 and env_type == 'gym':
                env.render()
            # select an action
            action = agent.act(state, eps)
            # take action in environment
            next_state, reward, done = env_step(env, action, env_type, brain_name)
            # update agent with returned information
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                total_steps += t
                break

        eps = max(eps_end, eps_decay*eps) # decrease epsilon

        # update stats
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        avg_score = np.mean(scores_window)
        avg_scores.append(avg_score)
        buffer_len = len(agent.memory)
        if avg_score > best_avg_score:    # update best average score
            best_avg_score = avg_score

        # print stats
        print('\rEpisode {:6}\tAvg: {:.2f}\tBest: {:.2f}\tEps: {:.4f}\tBufferLen: {:6}\t⍺: {:.4f}'.format(i_episode, avg_score, best_avg_score, eps, buffer_len, agent.alpha), end="")
        if i_episode % 100 == 0:
            n_secs = int(time.time() - time_start)
            print('\rEpisode {:6}\tAvg: {:.2f}\tBest: {:.2f}\tEps: {:.4f}\tBufferLen: {:6}\t⍺: {:.4f}\tSteps: {:7}\tSecs: {:4}'.format(i_episode, avg_score, best_avg_score, eps, buffer_len, agent.alpha, total_steps, n_secs))
            torch.save(agent.qnetwork_local.state_dict(), '../checkpoints/episode.' + str(i_episode) + '.pth')
            time_start = time.time()
            total_steps = 0
        if avg_score >= solve_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, avg_score))
            torch.save(agent.qnetwork_local.state_dict(), '../checkpoints/solved.pth')
            break
    # play sound to signal training is finished
    play_sound('../libs/fanfare.wav')
    if graph_results:
        plot(scores, avg_scores, agent.loss_list, agent.entropy_list)


def initialize_env(env, env_type, brain_name=None):
    """ Initialize environment and return its initial state. """

    if env_type == 'gym':
        state = env.reset()
        #state = np.eye(64)[state] # one-hot encode for FrozenLake
    elif env_type == 'unity':
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
    return state


def env_step(env, action, env_type, brain_name=None):
    """ Given an action, return the state, reward, done. """

    if env_type == 'gym':
        state, reward, done, _ = env.step(action)
        #state, reward, done, _ = env.step([(action/2) - 2]) # action discretization for Pendulum
        #state = np.eye(64)[state] # one-hot encode for FrozenLake
    elif env_type == 'unity':
        env_info = env.step(action)[brain_name]        # send the action to the environment
        state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]
    return state, reward, done


def play_sound(file):
    """ Play a sound. """

    wave_obj = sa.WaveObject.from_wave_file(file)
    play_obj = wave_obj.play()
    play_obj.wait_done()

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


def watch(env, agent, checkpoints, frame_sleep=0.05, env_type='gym', brain_name=None):
    """ Visualize agent using saved checkpoints. """

    for checkpoint in checkpoints:
        # load saved weights from file
        print('Watching: {}'.format(checkpoint))
        agent.qnetwork_local.load_state_dict(torch.load('../checkpoints/' + checkpoint + '.pth'))
        # initialize environment
        state = initialize_env(env, env_type, brain_name)
        # interact with environment
        for _ in range(600):
            # slect an action
            action = agent.act(state)
            time.sleep(frame_sleep)
            # take action in environment
            state, _, done = env_step(env, action, env_type, brain_name)
            if env_type == 'gym':
                env.render()
            if done:
                break
