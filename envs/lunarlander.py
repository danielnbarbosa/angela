import sys
sys.path.insert(0, '../libs')
from monitor import train, watch
from dqn_agent import Agent
from environment import Environment


environment = Environment('LunarLander-v2', 'gym')
agent = Agent(state_size=8, action_size=4, fc1_units=64, fc2_units=64, seed=0, double_dqn=True, model='dueling')
train(environment, agent, n_episodes=4000, max_t=2000, solve_score=200.0)

# visualize agent training
#checkpoints = ['lunarlander']
#watch(env, agent, checkpoints, frame_sleep=0.0)
