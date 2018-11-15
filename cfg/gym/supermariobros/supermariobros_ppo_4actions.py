algorithm='ppo'
env_class='GymMario'
model_class='TwoLayerConv2D'

environment = {
    'name': 'ppaquette/meta-SuperMarioBros-Tiles-v0'
}

model = {
    'state_size': (4, 13, 16),
    'action_size': 4,
    'filter_maps': (32, 32),
    'kernels': (3, 3),
    'strides': (2, 2),
    'conv_out': (2, 3),
    'fc_units': 256,
    'seed': 0
}

agent = {
    'lr': 0.0001,
    'action_map': {0: [0, 1, 0, 0, 0, 1],
                   1: [0, 0, 0, 1, 0, 1],
                   2: [0, 1, 0, 0, 1, 1],
                   3: [0, 0, 0, 1, 1, 1]}
}

train = {
    'n_episodes':   50000,
    'max_t':        10000,
    'action_repeat': 4,
    'epsilon':      0.2,
    'beta':         0.00,
    'sgd_epoch':    2,
    'n_trajectories': 2
}
