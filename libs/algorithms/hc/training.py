"""
Training loop.
"""

#import glob
#import os
import pickle
import statistics
import numpy as np
import libs.statistics


def train(environment, agent, seed=0, n_episodes=2000, max_t=1000,
          gamma=1.0,
          npop=1,
          render=False,
          solve_score=100000.0,
          graph_when_done=False):
    """ Run training loop for Hill Climbing.

    Params
    ======
        environment: environment object
        agent: agent object
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        npop (int): population size for steepest ascent
        render (bool): whether to render the agent
        solve_score (float): criteria for considering the environment solved
        graph_when_done (bool): whether to show matplotlib graphs of the training run
    """
    np.random.seed(seed)

    stats = libs.statistics.Stats()
    stats_format = 'Best: {:8.2f}   Noise: {:6.4f}'

    # remove checkpoints from prior run
    #prior_checkpoints = glob.glob('checkpoints/last_run/episode*.pck')
    #for checkpoint in prior_checkpoints:
    #    os.remove(checkpoint)

    for i_episode in range(1, n_episodes+1):
        # generate noise for each member of population
        pop_noise = np.random.randn(npop, *agent.model.weights.shape)
        # generate placeholders for each member of population
        pop_return = np.zeros(npop)
        pop_rewards = []
        # rollout one episode for each population member and gather the rewards
        for j in range(npop):
            rewards = []
            # evaluate each population member
            agent.model.weights = agent.max_best_weights + agent.noise_scale * pop_noise[j]
            state = environment.reset()
            for t in range(max_t):
                if render:  # optionally render agent
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


        # every episode
        pop_best_rewards, pop_best_return = agent.learn(pop_noise, pop_return, pop_rewards)
        stats.update(len(pop_best_rewards), pop_best_rewards, i_episode)
        stats.print_episode(i_episode, t, stats_format, agent.max_best_return, agent.noise_scale)

        # every epoch (100 episodes)
        if i_episode % 100 == 0:
            stats.print_epoch(i_episode, stats_format, agent.max_best_return, agent.noise_scale)
            save_name = 'checkpoints/last_run/episode.{}.pck'.format(i_episode)
            pickle.dump(agent.model.weights, open(save_name, 'wb'))

        # if solved
        if stats.is_solved(i_episode, solve_score):
            stats.print_solve(i_episode, stats_format, agent.max_best_return, agent.noise_scale)
            agent.model.weights = agent.max_best_weights
            pickle.dump(agent.model.weights, open('checkpoints/last_run/solved.pck', 'wb'))
            break

    # training finished
    if graph_when_done:
        stats.plot()
