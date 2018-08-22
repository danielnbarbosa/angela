from imports import *

environment = GymEnvironment('LunarLander-v2')

agent = Agent(state_size=8, action_size=4, fc1_units=64, fc2_units=64, seed=0,
              use_double_dqn=True,
              use_prioritized_experience_replay=False)

train(environment, agent, n_episodes=4000, max_t=2000, solve_score=200.0)

# visualize agent training
#checkpoints = ['lunarlander']
#watch(environment, agent, checkpoints, frame_sleep=0.0)
