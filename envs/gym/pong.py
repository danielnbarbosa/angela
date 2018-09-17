from imports import *

SEED = 0
#SEED = random.randint(0, 2 ** 30)

environment = environments.GymAtari('Pong-v0', seed=SEED)


def pg(render, load_file):
    model = models.PGConv2D(state_size=(4, 80, 80), action_size=3, fc1_units=200, seed=SEED)
    agent = agents.PolicyGradient(model, seed=SEED,
                                  lr=0.0001,
                                  load_file=load_file,
                                  action_map={0: 0, 1: 2, 2: 5})
    train_pg(environment, agent, n_episodes=50000, max_t=10000,
             gamma=0.99,
             render=render,
             graph_when_done=True)
