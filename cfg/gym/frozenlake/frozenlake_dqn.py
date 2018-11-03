algorithm='dqn'
env_class='Gym'
model_class='Dueling2x'

environment = {
    'name': 'FrozenLake-v0',
    'one_hot': 16
}

model = {
    'state_size': 16,
    'action_size': 4,
    'fc_units': (32, 32)
}

agent = {
    'action_size': 4,
    'use_double_dqn': True
}

train = {
    'n_episodes': 6000,
    'max_t': 1000
}
