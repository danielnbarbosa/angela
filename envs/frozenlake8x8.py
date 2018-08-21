import gym
import sys
sys.path.insert(0, '../libs')
from monitor import train, watch
from dqn_agent import Agent
from environment import Environment


environment = Environment('FrozenLake8x8-v0', 'gym', one_hot=64)
agent = Agent(state_size=64, action_size=4, fc1_units=64, fc2_units=64, seed=0)
train(environment, agent, n_episodes=4000, max_t=1000)
