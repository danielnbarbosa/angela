algorithm='pg'
env_class='GymAtariBreakout'
model_class='TwoLayerConv2D'

environment = {
    'name': 'Breakout-v0'
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
    'action_map': {0: 1, 1: 2, 2: 3}
}

train = {
    'n_episodes': 50000,
    'max_t': 10000,
}
