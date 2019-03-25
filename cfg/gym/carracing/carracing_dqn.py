algorithm='dqn'
env_class='GymCarRacing'
model_class='Conv2D2x'

environment = {
    'name': 'CarRacing-v0'
}

model = {
    'state_size': (4, 96, 96),
    'action_size': 4
}

agent = {
    'action_size': 4,
    'batch_size': 32,
    'use_double_dqn': False,
    'gamma': 0.99,
    'update_every': 4,
    'lr': 0.0006,
    'buffer_size': 100_000,
    'action_map': {0: [0., 1., 0.],
                   1: [0., 0., 1.],
                   2: [-1., 0., 0.],
                   3: [1., 0., 0.]}
}

train = {
    'n_episodes': 100_000,
    'max_t': 1000,
    'solve_score': 1000.0,
    'eps_start': 0.9,
    'eps_end': 0.1,
    'eps_decay': 0.9995
}
