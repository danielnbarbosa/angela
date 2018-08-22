from imports import *


environment = GymEnvironment('FrozenLake-v0', one_hot=16)

agent = Agent(state_size=16, action_size=4, fc1_units=32, fc2_units=32, seed=0,
              use_double_dqn=True,
              use_prioritized_experience_replay=False)

train(environment, agent, n_episodes=4000, max_t=1000)
