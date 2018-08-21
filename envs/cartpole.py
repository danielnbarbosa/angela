import sys
sys.path.insert(0, '../libs')
from monitor import train, watch
from dqn_agent import Agent
from environment import Environment


environment = Environment('CartPole-v1', 'gym')
agent = Agent(state_size=4, action_size=2, fc1_units=64, fc2_units=32, seed=0)
train(environment, agent, n_episodes=1000, solve_score=195.0)


# visualize agent training
#checkpoints = ['cartpole']
#watch(environment, agent, checkpoints, frame_sleep=0.0)
