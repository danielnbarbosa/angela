from imports import *

"""
NOTE: need to make some modifications to support multiple simultaneous actions
"""

environment = GymEnvironment('BipedalWalker-v2', action_bins=(5,5,5,5))

model = DuelingQNet(state_size=24, action_size=256, fc1_units=64, fc2_units=64, seed=0)

agent = Agent(model, state_size=24, action_size=256,
              use_double_dqn=True,
              use_prioritized_experience_replay=False)

train(environment, agent, n_episodes=4000, max_t=1000)
