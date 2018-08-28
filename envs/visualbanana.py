from imports import *

"""
NOTE: download pre-built Unity VisualBannana.app from: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/
"""

SEED=42
#SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = UnityMLEnvironment('VisualBanana.app', 'visual', seed=SEED)
#environment = UnityMLEnvironment('VisualBanana_Linux/Banana.x86_64', 'visual', seed=SEED)

#model = ThreeDConvQNet(state_size=(3, 4, 42, 42), action_size=4, seed=SEED)
model = ConvQNet(state_size=(1, 42, 42), action_size=4, seed=SEED)


agent = Agent(model, action_size=4, seed=SEED,
              use_double_dqn=True,
              use_prioritized_experience_replay=False)

# don't forget to reset epsilon when continuing training
#load(model, 'latest.pth')
train(environment, agent, n_episodes=10000, solve_score=13.0,
      eps_start=1.0,
      eps_end=0.05,
      eps_decay=0.997)
