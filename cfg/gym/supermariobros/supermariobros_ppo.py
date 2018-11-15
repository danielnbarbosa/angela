algorithm='ppo'
env_class='GymMario'
model_class='TwoLayerConv2D'

environment = {
    'name': 'ppaquette/meta-SuperMarioBros-Tiles-v0'
}

model = {
    'state_size': (4, 13, 16),
    'action_size': 14,
    'filter_maps': (32, 32),
    'kernels': (3, 3),
    'strides': (2, 2),
    'conv_out': (2, 3),
    'fc_units': 256,
    'seed': 0
}

agent = {
    'lr': 0.0001,
    'action_map': {0: [0, 0, 0, 0, 0, 0],
                   1: [1, 0, 0, 0, 0, 0],
                   2: [0, 1, 0, 0, 0, 0],
                   3: [0, 0, 1, 0, 0, 0],
                   4: [0, 0, 0, 1, 0, 0],
                   5: [0, 0, 0, 0, 1, 0],
                   6: [0, 0, 0, 0, 0, 1],
                   7: [0, 0, 0, 0, 1, 1],
                   8: [0, 1, 0, 0, 1, 0],
                   9: [0, 0, 0, 1, 1, 0],
                   10: [0, 1, 0, 0, 0, 1],
                   11: [0, 0, 0, 1, 0, 1],
                   12: [0, 1, 0, 0, 1, 1],
                   13: [0, 0, 0, 1, 1, 1]}
}

train = {
    'n_episodes':   50000,
    'max_t':        10000,
    'epsilon':      0.1,
    'beta':         0.00,
    'sgd_epoch':    4,
    'n_trajectories': 2
}
