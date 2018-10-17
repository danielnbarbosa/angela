algorithm='pg'
env_class='GymAtariPong'
model_class='TwoLayerConv2D'

environment = {
    'name': 'PongDeterministic-v4',
}

model = {
    'state_size': (4, 80, 80),
    'action_size': 3,
    'filter_maps': (16, 32),
    'kernels': (8, 4),
    'strides': (4, 3),
    'conv_out': 6,
    'fc_units': 200
}

agent = {
    'lr': 0.0001,
    'action_map': {0: 0, 1: 4, 2: 5}
}

train = {
    'n_episodes': 50000,
    'max_t': 10000,
    'max_noop': 30
}
