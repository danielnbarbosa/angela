import gym
import sys
sys.path.insert(0, '../libs')
from monitor import train, watch
from dqn_agent import Agent
from environment import Environment


environment = Environment('FrozenLake-v0', 'gym', one_hot=16)
agent = Agent(state_size=16, action_size=4, fc1_units=32, fc2_units=32, seed=0)
train(environment, agent, n_episodes=4000, max_t=1000)
