from imports import *

SEED = 0
#SEED = random.randint(0, 2 ** 30)

environment = environments.GymAtari('Breakout-v0', seed=SEED)


def pg(render, load_file):
    model = models.PGConv2DBig(state_size=(4, 80, 80), action_size=4, fc_units=512, seed=SEED)
    agent = agents.PolicyGradient(model, seed=SEED,
                                  lr=0.0001,
                                  load_file=load_file)
    train_pg(environment, agent, n_episodes=10000, max_t=10000,
             gamma=0.99,
             render=render,
             graph_when_done=False)
