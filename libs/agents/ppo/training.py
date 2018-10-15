"""
Training loop.
"""

#import glob
#import os
import random
import torch
import libs.statistics


def train(environment, agent, seed=0, n_episodes=10000, max_t=2000,
          gamma=0.99,
          max_noop=0,
          epsilon=0.1,
          beta=.01,
          sgd_epoch=4,
          n_trajectories=1,
          render=False,
          solve_score=100000.0,
          graph_when_done=False):
    """ Run training loop for Policy Gradients.

    Params
    ======
        environment: environment object
        agent: agent object
        seed (int): Random seed
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        max_noop (int): maximum number of initial noops at start of episode
        epsilon (float): PPO clipping parameter
        beta (float): scaling factor for entropy
        sgd_epoch (int): number of times to run gradient descent using current gradients
        n_trajectories (int): number of trajectories to gather under same policy
        render (bool): whether to render the agent
        solve_score (float): criteria for considering the environment solved
        graph_when_done (bool): whether to show matplotlib graphs of the training run
    """
    random.seed(seed)
    stats = libs.statistics.PPOStats()
    stats_format = 'ε: {:6.4}   β:   {:6.4}'

    # remove checkpoints from prior run
    #prior_checkpoints = glob.glob('checkpoints/last_run/episode*.pth')
    #for checkpoint in prior_checkpoints:
    #    os.remove(checkpoint)

    for i_episode in range(1, n_episodes+1):
        old_probs = []
        states = []
        actions = []
        rewards_lists = []  # list of lists
        # take a random amount of noop actions before starting episode to inject stochasticity
        #for i in range(random.randint(0, max_noop)):
        #    if render:  # optionally render agent
        #        environment.render()
        #    state, reward, done = environment.step(0)
        # collect trajectories
        for _ in range(n_trajectories): # collect a few different trajectories under the same policy
            rewards = []
            state = environment.reset()
            for t in range(max_t):
                if render:  # optionally render agent
                    environment.render()
                action, action_index, prob = agent.act(state)
                next_state, reward, done = environment.step(action)
                # append to lists
                old_probs.append(prob)
                states.append(state)
                actions.append(action_index)
                rewards.append(reward)
                state = next_state
                if done:
                    break
            # append rewards from trajectory to rewards_lists
            # needs special treatment because rewards are normalized per trajectory
            rewards_lists.append(rewards)

        # every episode
        for _ in range(sgd_epoch):
            agent.learn(old_probs, states, actions, rewards_lists, gamma, epsilon, beta)
        # average rewards across all trajectories for stats
        flatten = lambda l: [item for sublist in l for item in sublist]
        rewards = flatten(rewards_lists)
        rewards = [r/n_trajectories for r in rewards]
        stats.update(t, rewards, i_episode)
        stats.print_episode(i_episode, t, stats_format, epsilon, beta)
        epsilon *= 0.999  # decay the clipping parameter
        beta *= 0.995  # decay the entropy, this reduces exploration in later runs

        # every epoch (100 episodes)
        if i_episode % 100 == 0:
            stats.print_epoch(i_episode, stats_format, epsilon, beta)
            save_name = 'checkpoints/last_run/episode.{}.pth'.format(i_episode)
            torch.save(agent.model.state_dict(), save_name)

        # if solved
        if stats.is_solved(i_episode, solve_score):
            stats.print_solve(i_episode, stats_format, epsilon, beta)
            torch.save(agent.model.state_dict(), 'checkpoints/last_run/solved.pth')
            break

    # training finished
    if graph_when_done:
        stats.plot()
