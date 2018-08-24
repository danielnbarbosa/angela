from imports import *

"""
NOTE: download pre-built Unity VisualBannana.app from: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/
"""

environment = UnityMLEnvironment('VisualBanana.app', 'visual')
#environment = UnityMLEnvironment('VisualBanana_Linux/Banana.x86_64', 'visual')

model = ConvQNet(state_size=(3, 84, 84), action_size=4, seed=0)

agent = Agent(model, action_size=4,
              use_double_dqn=True,
              use_prioritized_experience_replay=False)

# don't forget to reset epsilon when continuing training
#load(model, 'visualbanana.pth')
train(environment, agent, n_episodes=10000, solve_score=13.0,
      eps_start=1.0,
      eps_end=0.01,
      eps_decay=0.997)
