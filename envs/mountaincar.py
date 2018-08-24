from imports import *


environment = GymEnvironment('MountainCar-v0', max_steps=1000)

model = DuelingQNet(state_size=2, action_size=3, fc1_units=64, fc2_units=64, seed=0)

agent = Agent(model, action_size=3,
              use_double_dqn=True,
              use_prioritized_experience_replay=False)

#load(model, 'mountaincar.pth')
train(environment, agent, n_episodes=4000, max_t=1000, solve_score=-110.0)
