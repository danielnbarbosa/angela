from imports import *

SEED = 0
#SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = environments.Gym('Acrobot-v1', seed=SEED)


def dqn():
    model = models.DQNDueling_Q(state_size=6, action_size=3, fc_units=(64, 64), seed=SEED)
    agent = agents.DQN(model, action_size=3, seed=SEED,
                     use_double_dqn=True,
                     use_prioritized_experience_replay=False)
    train_dqn(environment, agent, n_episodes=12000, max_t=1000)


def hc():
    agent = HillClimbingAgent(state_size=6, action_size=3, seed=SEED, policy='deterministic')
    train_hc(environment, agent, seed=SEED, n_episodes=1000, max_t=1000,
             use_adaptive_noise=True,
             npop=5,
             noise_scale_in=2,
             noise_scale_out=2,
             print_every=10,
             render_every=10000,
             graph_when_done=True)


def pg():
    model = models.PGOneHiddenLayer(state_size=6, action_size=3, fc1_units=32, seed=SEED)
    agent = PolicyGradientAgent(model, seed=SEED, lr=0.005)
    train_pg(environment, agent, n_episodes=4000, max_t=1000,
             gamma=0.99,
             graph_when_done=True)
