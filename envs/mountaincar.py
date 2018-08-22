from imports import *


environment = Environment('MountainCar-v0', 'gym', max_steps=1000)

agent = Agent(state_size=2, action_size=3, fc1_units=64, fc2_units=64, seed=0,
              use_double_dqn=True,
              use_prioritized_experience_replay=False)

train(environment, agent, n_episodes=4000, max_t=1000, solve_score=-110.0)
