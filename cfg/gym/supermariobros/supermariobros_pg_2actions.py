algorithm='pg'
env_class='GymMario'
model_class='TwoLayerConv2D'

environment = {
    'name': 'ppaquette/meta-SuperMarioBros-Tiles-v0',
}

model = {
    'state_size': (4, 13, 16),
    'action_size': 4,
    'filter_maps': (32, 32),
    'kernels': (3, 3),
    'strides': (2, 2),
    'conv_out': (2, 3),
    'fc_units': 128,
    'normalize': 4,
}

agent = {
    'lr': 0.0003,
    'action_map': {0: [0, 0, 0, 1, 0, 1],
                   1: [0, 0, 0, 1, 1, 1],
                   2: [0, 1, 0, 0, 0, 1],
                   3: [0, 1, 0, 0, 1, 1]}
}

train = {
    'n_episodes':   50000,
    'max_t':        10000,
    'action_repeat': 2,
}
