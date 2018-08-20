import gym
import sys
sys.path.insert(0, '../libs')
from dqn_agent import Agent
from monitor import train, watch

N_MAX_STEPS = 1000

env = gym.make('Pendulum-v0')
env._max_episode_steps = N_MAX_STEPS

"""
uncomment action discretization in monitor.py
comment out `state, reward, done, _ = env.step(action)`
"""

agent = Agent(state_size=3, action_size=9, fc1_units=32, fc2_units=32, seed=0)
train(env, agent, n_episodes=4000, max_t=N_MAX_STEPS)
