from imports import *


"""
NOTE: download pre-built Unity Bannana.app from: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/
"""

#environment = UnityMLEnvironment('Banana.app', 'vector')
environment = UnityMLEnvironment('Banana_Linux/Banana.x86_64', 'vector')

model = TwoHiddenLayerQNet(state_size=37, action_size=4, fc1_units=32, fc2_units=32, seed=0)

agent = Agent(model, action_size=4,
              use_double_dqn=False,
              use_prioritized_experience_replay=False)

train(environment, agent, n_episodes=1000, solve_score=13.0,
      eps_start=1.0,
      eps_end=0.001,
      eps_decay=0.97)
