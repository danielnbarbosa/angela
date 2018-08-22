from imports import *

environment = GymEnvironment('CartPole-v1')

model = DuelingQNet(state_size=4, action_size=2, fc1_units=64, fc2_units=32, seed=0)

agent = Agent(model, action_size=2,
              use_double_dqn=True,
              use_prioritized_experience_replay=True,
              alpha_start=0.5,
              alpha_decay=0.9992)

train(environment, agent, n_episodes=1000, max_t=1000, solve_score=195.0)


# visualize agent training
#checkpoints = ['cartpole']
#watch(environment, agent, checkpoints, frame_sleep=0.0)
