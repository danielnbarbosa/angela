"""
Functions to execute training and visualize agent.
"""

import glob
import os
import time
from collections import deque
import numpy as np
import pickle
import torch
from visualize import plot_dqn, plot_hc, plot_pg, show_frames
import statistics

def train_pg(environment, agent, seed, n_episodes=10000, max_t=2000,
             gamma=1.0,
             print_every=100,
             render_every=100000,
             solve_score=100000.0,
             graph_when_done=False):

    stats = statistics.Stats()

    # remove checkpoints from prior run
    #prior_checkpoints = glob.glob('checkpoints/last_run/episode*.pth')
    #for checkpoint in prior_checkpoints:
    #    os.remove(checkpoint)

    for i_episode in range(1, n_episodes+1):
        saved_log_probs = []
        rewards = []
        state = environment.reset()
        for t in range(max_t):
            # render during training
            if i_episode % render_every == 0:
                environment.render()
            action, log_prob = agent.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done = environment.step(action)
            rewards.append(reward)
            if done:
                break

        # every episode
        agent.learn(rewards, saved_log_probs, gamma)
        stats.update(t, rewards, i_episode)
        stats.print_pg_episode(i_episode)

        # every epoch (100 episodes)
        if i_episode % 100 == 0:
            stats.print_pg_epoch(i_episode)
            save_name = 'checkpoints/last_run/episode.{}.pth'.format(i_episode)
            torch.save(agent.model.state_dict(), save_name)

        # if solved
        if stats.is_solved(i_episode, solve_score):
            stats.print_pg_solve(i_episode)
            torch.save(agent.model.state_dict(), 'checkpoints/last_run/solved.pth')
            break

    # training finished
    #if sound_when_done:
    #    play_sound('libs/fanfare.wav')
    if graph_when_done:
        plot_pg(scores, avg_scores)



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

    stats = statistics.Stats()
    best_return = -np.Inf               # current best return
    best_weights = agent.weights        # current best weights

    # remove checkpoints from prior run
    #prior_checkpoints = glob.glob('checkpoints/last_run/episode*.pck')
    #for checkpoint in prior_checkpoints:
    #    os.remove(checkpoint)

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


        # every episode
        # TODO: suff some of the above into agent.learn()
        stats.update(len(pop_best_rewards), pop_best_rewards, i_episode)
        stats.print_hc_episode(i_episode, pop_best_return, best_return, noise_scale)

        # every epoch (100 episodes)
        if i_episode % 100 == 0:
            stats.print_hc_epoch(i_episode, pop_best_return, best_return, noise_scale)
            save_name = 'checkpoints/last_run/episode.{}.pck'.format(i_episode)
            pickle.dump(agent.weights, open(save_name, 'wb'))

        # if solved
        if stats.is_solved(i_episode, solve_score):
            stats.print_hc_solve(i_episode, pop_best_return, best_return, noise_scale)
            agent.weights = best_weights
            pickle.dump(agent.weights, open('checkpoints/last_run/solved.pck', 'wb'))
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

    stats = statistics.Stats()
    eps = eps_start                     # initialize epsilon

    # remove checkpoints from prior run
    #prior_checkpoints = glob.glob('checkpoints/last_run/episode*.pth')
    #for checkpoint in prior_checkpoints:
    #    os.remove(checkpoint)

    for i_episode in range(1, n_episodes+1):
        rewards = []
        state = environment.reset()

        # loop over steps
        for t in range(max_t):
            # render during training
            if i_episode % render_every == 0:
                environment.render()

            # select an action
            action = agent.act(state, eps)
            # take action in environment
            next_state, reward, done = environment.step(action)
            # update agent with returned information
            agent.step(state, action, reward, next_state, done)
            state = next_state
            rewards.append(reward)
            if done:
                break

        # every episode
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        buffer_len = len(agent.memory)
        stats.update(t, rewards, i_episode)
        stats.print_dqn_episode(i_episode, eps, agent.alpha, buffer_len)

        # every epoch (100 episodes)
        if i_episode % 100 == 0:
            stats.print_dqn_epoch(i_episode, eps, agent.alpha, buffer_len)
            save_name = 'checkpoints/last_run/episode.{}.pth'.format(i_episode)
            torch.save(agent.qnetwork_local.state_dict(), save_name)

        # if solved
        if stats.is_solved(i_episode, solve_score):
            stats.print_dqn_solve(i_episode, eps, agent.alpha, buffer_len)
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoints/last_run/solved.pth')
            break

    # training finished
    #if sound_when_done:
    #    play_sound('libs/fanfare.wav')
    if graph_when_done:
        plot_dqn(scores, avg_scores, agent.loss_list, agent.entropy_list)


def watch(environment, agent, checkpoints, frame_sleep=0.05):
    """ Visualize agent using saved checkpoints. """

    for checkpoint in checkpoints:
        # load saved weights from file
        print('Watching: {}'.format(checkpoint))
        agent.qnetwork_local.load_state_dict(torch.load('checkpoints/' + checkpoint + '.pth'))
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


def load_dqn(model, file_name):
    """Load saved model weights from a checkpoint file for DQN agent."""

    model.local.load_state_dict(torch.load('checkpoints/' + file_name))
    model.target.load_state_dict(torch.load('checkpoints/' + file_name))
    print('Loaded: {}'.format(file_name))


def load_pickle(agent, file_name):
    """Load saved model weights from a pickle file for HC agent."""

    agent.weights = pickle.load(open('checkpoints/' + file_name, 'rb'))
    print('Loaded: {}'.format(file_name))


def load_model(model, file_name):
    """Load saved model weights from a checkpoint file for PG agent."""

    model.load_state_dict(torch.load('checkpoints/' + file_name))
    print('Loaded: {}'.format(file_name))
