algorithm='ppo'
env_class='GymAtariPong'
model_class='TwoLayerConv2D'

environment = {
    'name': 'Pong-v4'
}

model = {
    'state_size': (4, 80, 80),
    'action_size': 2,
    'filter_maps': (16, 32),
    'kernels': (6, 6),
    'strides': (2, 4),
    'conv_out': (9, 9),
    'fc_units': 256,
    'seed': 1
}

agent = {
    'lr': 0.0001,
    'action_map': {0: 4, 1: 5}
}

train = {
    'n_episodes':   50000,
    'max_t':        2000,
    'epsilon':      0.2,
    'beta':         0.02,
    'sgd_epoch':    4,
    'n_trajectories': 8
}
