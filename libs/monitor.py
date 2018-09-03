"""
Functions to execute training and visualize agent.
"""

import glob
import os
import time
from collections import deque
import numpy as np
import torch
from visualize import plot_dqn, plot_hc


def train_hc(environment, agent, seed, n_episodes=2000, max_t=1000,
             gamma=1.0,
             noise_scale=1e-2,
             use_adaptive_noise=False,
             noise_scale_in=2,
             noise_scale_out=2,
             noise_min=1e-3,
             noise_max=2,
             npop=1,
             print_every=100,
             render_every=100000,
             solve_score=100000.0,
             sound_when_done=False,
             graph_when_done=False):
    """ Run training loop for Hill Climbing.

    Params
    ======
        environment: environment object
        agent: agent object
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        noise_scale (float): standard deviation of additive noise
        use_adaptive_noise (bool): whether to implement adaptive noise
        noise_scale_in (int): factor to reduce noise by
        noise_scale_out (int): factor to increase noise by, set to 1 for simmulated annealing
        noise_min (int): minimum noise_scale
        noise_max (int): maximum noise scale
        npop (int): population size for steepest ascent
        render_every (int): render the agent interacting in the environment every n episodes
        solve_score (float): criteria for considering the environment solved
        sound_when_done (bool): wheter to play a sound to announce training is finished
        graph_when_done (bool): whether to show matplotlib graphs of the training run
    """
    np.random.seed(seed)

    scores = []                         # list containing scores from each episode
    avg_scores = []                     # list containing average scores after each episode
    scores_window = deque(maxlen=100)   # last 100 scores
    best_avg_score = -np.Inf            # best score for a single episode
    best_return = -np.Inf               # current best return
    best_weights = agent.weights        # current best weights


    for i_episode in range(1, n_episodes+1):
        # generate noise for each member of population
        pop_noise = np.random.randn(npop, *agent.weights.shape)
        # generate placeholders for each member of population
        pop_return = np.zeros(npop)
        pop_rewards = []
        # rollout one episode for each population member and gather the rewards
        for j in range(npop):
            rewards = []
            # evaluate each population member
            agent.weights = best_weights + noise_scale * pop_noise[j]
            state = environment.reset()
            for t in range(max_t):
                # render during training
                if i_episode % render_every == 0:
                    environment.render()
                action = agent.act(state)
                state, reward, done = environment.step(action)
                rewards.append(reward)
                if done:
                    break
            # calculate return
            discounts = [gamma**i for i in range(len(rewards)+1)]
            pop_return[j] = sum([a*b for a, b in zip(discounts, rewards)])
            pop_rewards.append(rewards)

        # determine who got the highest reward
        pop_best_return = pop_return.max()

        # compare best return from current population to global best return
        if pop_best_return >= best_return: # found better weights
            best_return = pop_best_return
            best_weights += noise_scale * pop_noise[pop_return.argmax()]
            noise_scale = max(noise_min, noise_scale / noise_scale_in) if use_adaptive_noise else noise_scale
        else: # did not find better weights
            noise_scale = min(noise_max, noise_scale * noise_scale_out) if use_adaptive_noise else noise_scale

        # consider the best rewards from the current population for calculating stats
        pop_best_rewards = pop_rewards[pop_return.argmax()]

        # update stats
        scores_window.append(sum(pop_best_rewards))
        scores.append(sum(pop_best_rewards))
        avg_score = np.mean(scores_window)
        avg_scores.append(avg_score)
        # update best average score, let a few episodes pass in case you get lucky early
        if avg_score > best_avg_score and i_episode > 30:
            best_avg_score = avg_score

        # print stats every n episodes
        if i_episode % print_every == 0:
            print('\rEpisode {:5}\tAvg: {:5.2f}\tBestAvg: {:7.2f}\tCurRet: {:8.2f}\tBestRet: {:5.2f}\tNoise: {:.4f}'
                  .format(i_episode, avg_score, best_avg_score, pop_best_return, best_return, noise_scale))

        # if solved
        if avg_score >= solve_score and i_episode >= 100:
            print('\nEnvironment solved in {:d} episodes!\tAvgScore: {:.3f}\tStdDev: {:.3f}\tSeed: {:d}'
                  .format(i_episode-100, avg_score, np.std(scores_window), environment.seed))
            agent.weights = best_weights
            break

    # training finished
    if graph_when_done:
        plot_hc(scores, avg_scores)



def train_dqn(environment, agent, n_episodes=2000, max_t=1000,
              eps_start=1.0,
              eps_end=0.01,
              eps_decay=0.995,
              render_every=100000,
              solve_score=100000.0,
              sound_when_done=False,
              graph_when_done=False):
    """ Run training loop for DQN.

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
    best_avg_score = -np.Inf            # best score for a single episode
    time_start = time.time()            # track wall time over 100 episodes
    total_steps = 0                     # track steps taken over 100 episodes

    # remove checkpoints from prior run
    prior_checkpoints = glob.glob('../../checkpoints/last_run/episode*.pth')
    for checkpoint in prior_checkpoints:
        os.remove(checkpoint)

    for i_episode in range(1, n_episodes+1):
        # reset environment
        state = environment.reset()
        score = 0
        # loop over steps
        for t in range(max_t):
            # render during training
            if i_episode % render_every == 0:
                environment.render()

            # visualize each frame of Conv3D state
            # show_frames()

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
        print('\rEpisode {:5}\tAvg: {:7.3f}\tBestAvg: {:7.3f}'
              '\tε: {:.4f}  ⍺: {:.4f}  Buffer: {:6}  CurrRet: {:5}'
              .format(i_episode, avg_score, best_avg_score, eps, agent.alpha, buffer_len, score), end="")

        # every 100 episodes
        if i_episode % 100 == 0:
            # calculate wall time
            n_secs = int(time.time() - time_start)
            # print extented stats
            print('\rEpisode {:5}\tAvg: {:7.3f}\tBestAvg: {:7.3f}'
                  '\tε: {:.4f}  ⍺: {:.4f}  Buffer: {:6}  Steps: {:6}  Secs: {:4}'
                  .format(i_episode, avg_score, best_avg_score, eps, agent.alpha, buffer_len, total_steps, n_secs))
            save_name = '../../checkpoints/last_run/episode.{}.pth'.format(i_episode)
            torch.save(agent.qnetwork_local.state_dict(), save_name)
            # reset counters
            time_start = time.time()
            total_steps = 0

        # if solved
        if avg_score >= solve_score and i_episode >= 100:
            print('\nEnvironment solved in {:d} episodes!\tAvgScore: {:.3f}\tStdDev: {:.3f}\tSeed: {:d}'
                  .format(i_episode-100, avg_score, np.std(scores_window), environment.seed))
            torch.save(agent.qnetwork_local.state_dict(), '../../checkpoints/last_run/solved.pth')
            break

    # training finished
    #if sound_when_done:
    #    play_sound('../../libs/fanfare.wav')
    if graph_when_done:
        plot_dqn(scores, avg_scores, agent.loss_list, agent.entropy_list)


def watch(environment, agent, checkpoints, frame_sleep=0.05):
    """ Visualize agent using saved checkpoints. """

    for checkpoint in checkpoints:
        # load saved weights from file
        print('Watching: {}'.format(checkpoint))
        agent.qnetwork_local.load_state_dict(torch.load('../../checkpoints/' + checkpoint + '.pth'))
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

    print('Loaded: {}'.format(file_name))
    model.local.load_state_dict(torch.load('../../checkpoints/' + file_name))
    model.target.load_state_dict(torch.load('../../checkpoints/' + file_name))
