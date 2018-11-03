algorithm='dqn'
env_class='Gym'
model_class='Dueling2x'

environment = {
    'name': 'Acrobot-v1',
}

model = {
    'state_size': 6,
    'action_size': 3,
    'fc_units': (64, 64)
}

agent = {
    'action_size': 3,
    'use_double_dqn': True
}

train = {
    'n_episodes': 12000,
    'max_t': 1000
}
