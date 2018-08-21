import sys
sys.path.insert(0, '../libs')
from monitor import train, watch
from dqn_agent import Agent
from environment import Environment


environment = Environment('BipedalWalker-v2', 'gym', action_bins(5,5,5,5))
agent = Agent(state_size=24, action_size=256, fc1_units=64, fc2_units=128, seed=0)
train(environment, agent, n_episodes=4000, max_t=1000)
