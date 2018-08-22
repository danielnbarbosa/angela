from imports import *

environment = GymEnvironment('FrozenLake8x8-v0', one_hot=64)

agent = Agent(state_size=64, action_size=4, fc1_units=64, fc2_units=64, seed=0,
              use_double_dqn=True,
              use_prioritized_experience_replay=False)

train(environment, agent, n_episodes=4000, max_t=1000)
