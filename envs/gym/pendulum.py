from imports import *

SEED = 0
#SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = environments.Gym('Pendulum-v0', seed=SEED, max_steps=1000, action_bins=(10,))


def dqn():
    model = models.DQNDueling_Q(state_size=3, action_size=9, fc_units=(32, 32), seed=SEED)
    agent = agents.DQN(model, action_size=9, seed=SEED,
                  use_double_dqn=True,
                  use_prioritized_experience_replay=False)
    train_dqn(environment, agent, n_episodes=4000, max_t=1000)


def hc():
    agent = agents.HillClimbing(state_size=3, action_size=9, seed=SEED, policy='deterministic')
    train_hc(environment, agent, seed=SEED, n_episodes=1000, max_t=1000,
             use_adaptive_noise=True,
             npop=10,
             print_every=100,
             graph_when_done=True)


def pg():
    model = models.PGOneHiddenLayer(state_size=3, action_size=9, fc1_units=24, seed=SEED)
    agent = agents.PolicyGradient(model, seed=SEED, lr=0.005)
    train_pg(environment, agent, seed=SEED, n_episodes=5000, max_t=1000,
             gamma=0.99,
             graph_when_done=True)
