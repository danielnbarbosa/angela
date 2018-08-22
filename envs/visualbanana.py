import sys
sys.path.insert(0, '../libs')
from monitor import train, watch
from dqn_agent import Agent
from environment import Environment


"""
NOTE: download pre-built Unity Bannana.app from: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/
"""

environment = Environment('VisualBanana.app', 'unity')
agent = Agent(state_size=37, action_size=4, fc1_units=32, fc2_units=32, seed=0, double_dqn=False, model='cnn')
train(environment, agent, n_episodes=1000, eps_start=1.0, eps_end=0.001, eps_decay=0.97, solve_score=13.0, graph_results=False)


# visualize agent training
#checkpoints = ['bananas']
#watch(environment, agent, checkpoints, frame_sleep=0.07)
