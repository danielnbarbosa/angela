import sys
sys.path.insert(0, '../libs')
from monitor import train, watch
from dqn_agent import Agent
from environment import Environment


environment = Environment('MountainCar-v0', 'gym', max_steps=1000)
agent = Agent(state_size=2, action_size=3, fc1_units=64, fc2_units=64, seed=0)
train(environment, agent, n_episodes=4000, solve_score=-110.0)
