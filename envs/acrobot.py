from imports import *

environment = Environment('Acrobot-v1', 'gym')

agent = Agent(state_size=6, action_size=3, fc1_units=64, fc2_units=64, seed=0,
              use_double_dqn=True,
              use_prioritized_experience_replay=False)

train(environment, agent, n_episodes=4000, max_t=1000)


# visualize agent training
#checkpoints = ['episode.100', 'episode.200', 'episode.300']
#watch(environment, agent, checkpoints, frame_sleep=0.0)
