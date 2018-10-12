"""
Training loop.
"""

#import glob
#import os
import numpy as np
import torch
import libs.statistics


def train(environment, agent, n_episodes=2000, max_t=1000,
          render=False,
          solve_score=100000.0,
          graph_when_done=False):
    """ Run training loop for DQN.

    Params
    ======
        environment: environment object
        agent: agent object
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        render (bool): whether to render the agent
        solve_score (float): criteria for considering the environment solved
        graph_when_done (bool): whether to show matplotlib graphs of the training run
    """

    if agent.n_agents == 1:
        stats = libs.statistics.DeepDeterministicPolicyGradientStats()
    else:
        stats = libs.statistics.MultiAgentDeepDeterministicPolicyGradientStats()

    # remove checkpoints from prior run
    #prior_checkpoints = glob.glob('checkpoints/last_run/episode*.pth')
    #for checkpoint in prior_checkpoints:
    #    os.remove(checkpoint)

    for i_episode in range(1, n_episodes+1):
        rewards = []
        alive_agents = [True] * agent.n_agents  # list of agents that are still alive (not done)
        state = environment.reset()
        # loop over steps
        for t in range(max_t):
            if render:  # optionally render agent
                environment.render()
            # select an action
            if agent.inference_only:  # disable noise on inference
                action = agent.act(state, add_noise=False)
            else:
                action = agent.act(state)
            # take action in environment
            next_state, reward, done = environment.step(action)

            # set rewards to 0 for agents that are done
            for i, a in enumerate(alive_agents):
                reward[i] = reward[i] * a
            if True in done:
                indices = [i for i, d in enumerate(done) if d == True]
                for i in indices:
                    alive_agents[i] = False

            # update agent with returned information
            agent.step(state, action, reward, next_state, done)
            state = next_state
            rewards.append(reward)
            # DEBUG rewards and dones per step
            #print(alive_agents)
            #print(reward)
            #print(done)
            #input('->')
            if agent.n_agents == 1 and done:
                break
            if agent.n_agents > 1 and not any(alive_agents):
                break

        # every episode
        buffer_len = len(agent.memory)
        if agent.n_agents == 1:
            stats.update(t, rewards, i_episode)
            stats.print_episode(i_episode, agent.alpha, buffer_len, t)
        else:
            # DEBUG non zero rewards
            #flatten = lambda l: [item for sublist in l for item in sublist if item != 0.0]
            #print(' '.join('%5.2f' % reward for reward in flatten(rewards)))
            per_agent_rewards = []
            for i in range(agent.n_agents):
                per_agent_reward = 0
                for step in rewards:
                    per_agent_reward += step[i]
                per_agent_rewards.append(per_agent_reward)
            stats.update(t, [np.mean(per_agent_rewards)], i_episode)
            stats.print_episode(i_episode, agent.alpha, buffer_len, t, per_agent_rewards)

        # every epoch (100 episodes)
        if i_episode % 100 == 0:
            stats.print_epoch(i_episode, agent.alpha, buffer_len)
            save_name = 'checkpoints/last_run/episode.{}'.format(i_episode)
            torch.save(agent.actor_local.state_dict(), save_name + '.actor.pth')
            torch.save(agent.critic_local.state_dict(), save_name + '.critic.pth')

        # if solved
        if stats.is_solved(i_episode, solve_score):
            stats.print_solve(i_episode, agent.alpha, buffer_len)
            torch.save(agent.actor_local.state_dict(), 'checkpoints/last_run/solved.actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoints/last_run/solved.critic.pth')
            break

    # training finished
    if graph_when_done:
        stats.plot(agent.loss_list)
