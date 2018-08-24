from imports import *

environment = GymEnvironment('Acrobot-v1')

model = DuelingQNet(state_size=6, action_size=3, fc1_units=64, fc2_units=64, seed=0)

agent = Agent(model, action_size=3,
              use_double_dqn=True,
              use_prioritized_experience_replay=False)

train(environment, agent, n_episodes=4000, max_t=1000)
