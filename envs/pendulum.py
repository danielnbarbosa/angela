from imports import *

environment = GymEnvironment('Pendulum-v0', max_steps=1000, action_bins=(10,))

model = DuelingQNet(state_size=3, action_size=9, fc1_units=32, fc2_units=32, seed=0)

agent = Agent(model, action_size=9,
              use_double_dqn=True,
              use_prioritized_experience_replay=False)

train(environment, agent, n_episodes=4000, max_t=1000)
