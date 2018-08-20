import gym
import sys
sys.path.insert(0, '../libs')
from dqn_agent import Agent
from monitor import train, watch

env = gym.make('FrozenLake8x8-v0')
#env.seed(0)

"""
NOTE: uncomment two instances of one-hot encoding in monitor.py
"""

agent = Agent(state_size=64, action_size=4, fc1_units=64, fc2_units=64, seed=0)
train(env, agent, n_episodes=4000, max_t=1000)
