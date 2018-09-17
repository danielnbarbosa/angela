from imports import *

SEED = 0
#SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = environments.GymAtari('Pong-v0', seed=SEED)


def pg():
    model = models.PGConv2D(state_size=(4, 80, 80), action_size=3, fc1_units=200, seed=SEED)
    agent = agents.PolicyGradient(model, seed=SEED, lr=0.0001, action_map={0: 0, 1: 2, 2: 5})
    load_model(model, 'best/pong_4frames_01.pth')
    train_pg(environment, agent, seed=SEED, n_episodes=10000, max_t=10000,
             gamma=0.99,
             render_every=1,
             graph_when_done=True)
