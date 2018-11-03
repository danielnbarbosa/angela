algorithm='dqn'
env_class='Gym'
model_class='Dueling2x'

environment = {
    'name': 'Pendulum-v0',
    'max_steps': 1000,
    'action_bins': (10,)
}

model = {
    'state_size': 3,
    'action_size': 9,
    'fc_units': (32, 32)
}

agent = {
    'action_size': 9,
    'use_double_dqn': True
}

train = {
    'n_episodes': 4000,
    'max_t': 1000
}
