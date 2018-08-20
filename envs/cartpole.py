import gym
import sys
sys.path.insert(0, '../libs')
from dqn_agent import Agent
from monitor import train, watch


env = gym.make('CartPole-v1')

agent = Agent(state_size=4, action_size=2, fc1_units=64, fc2_units=32, seed=0)
train(env, agent, n_episodes=300, solve_score=195.0)

# visualize agent training
#checkpoints = ['episode.100', 'episode.200', 'episode.300']
#watch(env, agent, checkpoints, frame_sleep=0.0)
