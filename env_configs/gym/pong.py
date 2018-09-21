from libs import environments, models, agents, train

SEED = 0
#SEED = random.randint(0, 2 ** 30)

#environment = environments.GymAtari('Pong-v0', seed=SEED)
environment = environments.GymAtariPong('Pong-v0', seed=SEED)


def pg(render, load_file):
    #model = models.pg.BigConv2D(state_size=(4, 80, 80), action_size=3, fc_units=512, seed=SEED)
    model = models.pg.TwoLayerConv2D(state_size=(4, 80, 80), action_size=3,
                                     filter_maps=(16, 32), kernels=(8, 4), strides=(4, 3),
                                     conv_out=6, fc_units=200, seed=SEED)
    #model = models.pg.SmallConv2D(state_size=(4, 80, 80), action_size=3, fc_units=256, seed=SEED, normalize=False)
    agent = agents.PolicyGradient(model, seed=SEED,
                                  lr=0.0001,
                                  load_file=load_file,
                                  action_map={0: 0, 1: 2, 2: 5})
    train.pg(environment, agent, n_episodes=50000, max_t=10000, render=render)
