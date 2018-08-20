import gym
import sys
sys.path.insert(0, '../libs')
from dqn_agent import Agent
from monitor import train, watch

env = gym.make('Acrobot-v1')

agent = Agent(state_size=6, action_size=3, fc1_units=64, fc2_units=64, seed=0)
train(env, agent)

# visualize agent training
#checkpoints = ['episode.100', 'episode.200', 'episode.300']
#watch(env, agent, checkpoints, frame_sleep=0.0)
