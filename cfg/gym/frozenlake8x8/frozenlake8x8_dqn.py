algorithm='dqn'
env_class='Gym'
model_class='Dueling2x'

environment = {
    'name': 'FrozenLake8x8-v0',
    'one_hot': 64
}

model = {
    'state_size': 64,
    'action_size': 4,
    'fc_units': (64, 64)
}

agent = {
    'action_size': 4,
    'use_double_dqn': True
}

train = {
    'n_episodes': 6000,
    'max_t': 1000
}
