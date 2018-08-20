import gym
import sys
sys.path.insert(0, '../libs')
from dqn_agent import Agent
from monitor import train, watch

N_MAX_STEPS = 1000   # default is 200

env = gym.make('MountainCar-v0')
env._max_episode_steps = N_MAX_STEPS

agent = Agent(state_size=2, action_size=3, fc1_units=64, fc2_units=64, seed=0)
train(env, agent, n_episodes=4000, max_t=N_MAX_STEPS, solve_score=-110.0)
