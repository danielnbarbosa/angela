from imports import *

"""
NOTE: Need to make some modifications to support multiple simultaneous actions.
"""

SEED = 0
#SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = environments.Gym('BipedalWalker-v2', seed=SEED, action_bins=(5,5,5,5))


def dqn():
    model = models.DQNDueling_Q(state_size=24, action_size=256, fc_units=(64, 64), seed=SEED)
    agent = agents.DQN(model, state_size=24, action_size=256, seed=SEED,
                     use_double_dqn=True,
                     use_prioritized_experience_replay=False)
    train_dqn(environment, agent, n_episodes=4000, max_t=1000)
