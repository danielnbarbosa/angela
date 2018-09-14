from imports import *

SEED = 0
#SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = GymEnvironmentAtari('Pong-v0', seed=SEED)


def pg():
    model = PGConv2D(state_size=(2, 80, 80), action_size=3, fc1_units=200, seed=SEED)
    agent = PolicyGradientAgent(model, state_size=(2, 80, 80), seed=SEED, lr=0.0001)
    train_pg(environment, agent, seed=SEED, n_episodes=4000, max_t=1000,
             gamma=0.99,
             graph_when_done=True)
