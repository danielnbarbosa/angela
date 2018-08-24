from imports import *

environment = GymEnvironment('LunarLander-v2')

model = DuelingQNet(state_size=8, action_size=4, fc1_units=64, fc2_units=64, seed=0)

agent = Agent(model, action_size=4,
              use_double_dqn=True,
              use_prioritized_experience_replay=False)

#load(model, 'lunarlander.pth')
train(environment, agent, n_episodes=4000, max_t=2000, solve_score=200.0)
