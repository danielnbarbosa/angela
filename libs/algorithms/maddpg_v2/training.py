"""
Training loop.
"""

import numpy as np
import torch
import libs.statistics


def train(environment, agent, n_episodes=10000, max_t=1000,
          render=False,
          solve_score=0.5):
    """
    Params
    ======
        environment: environment object
        agent: agent object
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        solve_score (float): criteria for considering the environment solved
    """


    stats = libs.statistics.MultiAgentDDPGv2Stats()
    stats_format = 'Buffer: {:6}   NoiseW: {:.4}'

    for i_episode in range(1, n_episodes+1):
        rewards = []
        state = environment.reset()
        # loop over steps
        for t in range(1, max_t+1):
            # select an action
            if agent.evaluation_only:  # disable noise on evaluation
                action = agent.act(state, add_noise=False)
            else:
                action = agent.act(state)
            # take action in environment
            next_state, reward, done = environment.step(action)
            # update agent with returned information
            agent.step(state, action, reward, next_state, done)
            state = next_state
            rewards.append(reward)
            if any(done):
                break

        # every episode
        buffer_len = len(agent.memory)
        per_agent_rewards = []  # calculate per agent rewards
        for i in range(agent.n_agents):
            per_agent_reward = 0
            for step in rewards:
                per_agent_reward += step[i]
            per_agent_rewards.append(per_agent_reward)
        stats.update(t, [np.max(per_agent_rewards)], i_episode)  # use max over all agents as episode reward
        stats.print_episode(i_episode, t, stats_format, buffer_len, agent.noise_weight,
                            agent.agents[0].critic_loss, agent.agents[1].critic_loss,
                            agent.agents[0].actor_loss, agent.agents[1].actor_loss,
                            agent.agents[0].noise_val, agent.agents[1].noise_val,
                            per_agent_rewards[0], per_agent_rewards[1])

        # every epoch (100 episodes)
        if i_episode % 100 == 0:
            stats.print_epoch(i_episode, stats_format, buffer_len, agent.noise_weight)
            save_name = 'checkpoints/episode.{}.'.format(i_episode)
            for i, save_agent in enumerate(agent.agents):
                torch.save(save_agent.actor_local.state_dict(), save_name + str(i) + '.actor.pth')
                torch.save(save_agent.critic_local.state_dict(), save_name + str(i) + '.critic.pth')

        # if solved
        if stats.is_solved(i_episode, solve_score):
            stats.print_solve(i_episode, stats_format, buffer_len, agent.noise_weight)
            save_name = 'checkpoints/solved.'
            for i, save_agent in enumerate(agent.agents):
                torch.save(save_agent.actor_local.state_dict(), save_name + str(i) + '.actor.pth')
                torch.save(save_agent.critic_local.state_dict(), save_name + str(i) + '.critic.pth')
            break
