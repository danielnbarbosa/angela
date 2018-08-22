from imports import *


environment = Environment('CartPole-v1', 'gym')

agent = Agent(state_size=4, action_size=2, fc1_units=64, fc2_units=32, seed=0,
              use_double_dqn=True,
              use_prioritized_experience_replay=True)

train(environment, agent, n_episodes=1000, max_t=1000, solve_score=195.0)


# visualize agent training
#checkpoints = ['cartpole']
#watch(environment, agent, checkpoints, frame_sleep=0.0)
