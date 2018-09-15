from imports import *

SEED = 0
#SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = environments.Gym('Breakout-ram-v0', seed=SEED, normalize=True)


def dqn():
    #model = DQNDueling_Q(state_size=128, action_size=4, fc_units=(256, 256), seed=SEED)
    #model = DQNTwoHiddenLayer_Q(state_size=256, action_size=4, fc_units=(256, 128), seed=SEED)
    model = models.DQNFourHiddenLayer_Q(state_size=128, action_size=4, fc_units=(128, 128, 64, 32), seed=SEED)
    agent = agents.DQN(model, action_size=4, seed=SEED,
              use_double_dqn=True,
              use_prioritized_experience_replay=False,
              buffer_size=100000)
    train_dqn(environment, agent, n_episodes=10000, max_t=10000,
              eps_start=1,
              eps_end=0.1,
              eps_decay=0.9993,
              graph_when_done=True,
              render_every=10000000)


def hc():
    agent = agents.HillClimbing(state_size=128, action_size=4, seed=SEED, policy='stochastic')
    train_hc(environment, agent, seed=SEED, n_episodes=4000, max_t=2000,
             use_adaptive_noise=True,
             npop=4,
             print_every=1,
             graph_when_done=True)
