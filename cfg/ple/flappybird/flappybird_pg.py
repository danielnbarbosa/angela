algorithm='pg'
env_class='PLEFlappyBird'
model_class='TwoLayerConv2D'

environment = {
    'pipe_gap': 150
}

model = {
    'state_size': (4, 80, 80),
    'action_size': 2,
    'filter_maps': (32, 64),
    'kernels': (8, 4),
    'strides': (4, 3),
    'conv_out': 6,
    'fc_units': 256,
    'normalize': True
}

agent = {
    'lr': 0.0001,
    'action_map': {0: None, 1: 119}
}

train = {
    'n_episodes': 500000,
    'max_t': 2000
}
