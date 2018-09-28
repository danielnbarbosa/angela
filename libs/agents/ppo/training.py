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
          SGD_epoch=4,
          sample_epoch=5,
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
        SGD_epoch (int): number of times to run gradient descent using current gradients
        sample_epoch (int): number of trajectories to gather under same policy
        render (bool): whether to render the agent
        solve_score (float): criteria for considering the environment solved
        graph_when_done (bool): whether to show matplotlib graphs of the training run
    """
    random.seed(seed)
    stats = libs.statistics.ProximalPolicyOptimizationStats()

    # remove checkpoints from prior run
    #prior_checkpoints = glob.glob('checkpoints/last_run/episode*.pth')
    #for checkpoint in prior_checkpoints:
    #    os.remove(checkpoint)

    for i_episode in range(1, n_episodes+1):
        # each feed is a list of lists of length sample_epoch
        old_probs_feed = []
        states_feed = []
        actions_feed = []
        rewards_feed = []
        # take a random amount of noop actions before starting episode to inject stochasticity
        for i in range(random.randint(0, max_noop)):
            if render:  # optionally render agent
                environment.render()
            state, reward, done = environment.step(0)
        # collect trajectories
        for _ in range(sample_epoch): # collect a few different trajectories under the same policy
            old_probs = []
            states = []
            actions = []
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
            # append to feeds
            old_probs_feed.append(old_probs)
            states_feed.append(states)
            actions_feed.append(actions)
            rewards_feed.append(rewards)

        # every episode
        for _ in range(SGD_epoch):
            agent.learn(old_probs_feed, states_feed, actions_feed, rewards_feed, gamma, epsilon, beta)
        stats.update(t, rewards, i_episode)
        stats.print_episode(i_episode, epsilon, beta, t)
        epsilon *= 0.999  # decay the clipping parameter
        beta *= 0.995  # decay the entropy, this reduces exploration in later runs

        # every epoch (100 episodes)
        if i_episode % 100 == 0:
            stats.print_epoch(i_episode, epsilon, beta)
            save_name = 'checkpoints/last_run/episode.{}.pth'.format(i_episode)
            torch.save(agent.model.state_dict(), save_name)

        # if solved
        if stats.is_solved(i_episode, solve_score):
            stats.print_solve(i_episode, epsilon, beta)
            torch.save(agent.model.state_dict(), 'checkpoints/last_run/solved.pth')
            break

    # training finished
    if graph_when_done:
        stats.plot()
