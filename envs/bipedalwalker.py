from imports import *

"""
NOTE: need to make some modifications to support multiple actions
"""

environment = Environment('BipedalWalker-v2', 'gym', action_bins=(5,5,5,5))

agent = Agent(state_size=24, action_size=256, fc1_units=64, fc2_units=64, seed=0,
              use_double_dqn=True,
              use_prioritized_experience_replay=False)

train(environment, agent, n_episodes=4000, max_t=1000)
