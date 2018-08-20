from unityagents import UnityEnvironment
import sys
sys.path.insert(0, '../libs')
from dqn_agent import Agent
from monitor import train, watch
import random


# need to manually set seed to ensure a random environment is initialized
SEED = random.randint(0, 2 ** 30)
env = UnityEnvironment(file_name="Banana.app", seed=SEED)
brain_name = env.brain_names[0]

"""
NOTE: download pre-built Unity Bannana.app from: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/
"""

agent = Agent(state_size=37, action_size=4, fc1_units=32, fc2_units=32, seed=0, double_dqn=False, model='classic')
#train(env, agent, env_type='unity', brain_name=brain_name, n_episodes=1000, eps_start=1.0, eps_end=0.001, eps_decay=0.97, solve_score=13.0, graph_results=False)


# visualize agent training
checkpoints = ['bananas']
watch(env, agent, checkpoints, env_type='unity', brain_name=brain_name, frame_sleep=0.07)

env.close()
