import sys
sys.path.insert(0, '../libs')
from monitor import train, watch
from dqn_agent import Agent
from environment import Environment


environment = Environment('Acrobot-v1', 'gym')
agent = Agent(state_size=6, action_size=3, fc1_units=64, fc2_units=64, seed=0)
train(environment, agent)


# visualize agent training
#checkpoints = ['episode.100', 'episode.200', 'episode.300']
#watch(environment, agent, checkpoints, frame_sleep=0.0)
