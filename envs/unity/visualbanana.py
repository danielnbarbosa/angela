from imports import *

"""
NOTE: download pre-built Unity VisualBannana.app from: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/
"""

SEED=42
#SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = UnityMLVisualEnvironmentSimple('compiled_unity_environments/VisualBanana.app', seed=SEED)
#environment = UnityMLVisualEnvironmentSimple('compiled_unity_environments/VisualBanana_Linux/Banana.x86_64', seed=SEED)

# shape is (m, c, f, h, w)
model = Simple3DConvQNet(state_size=(3, 4, 84, 84), action_size=4, seed=SEED)


agent = DQNAgent(model, action_size=4, seed=SEED,
                 buffer_size=20000,
                 gamma=0.95,
                 lr=9e-4,
                 use_double_dqn=True,
                 use_prioritized_experience_replay=False)

# don't forget to reset epsilon when continuing training
#load(model, 'latest.pth')
train_dqn(environment, agent, n_episodes=10000, solve_score=13.0,
          eps_start=1.0,
          eps_end=0.05,
          eps_decay=0.995)
