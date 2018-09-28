agent_type='ppo'
env_class='Gym'
model_class='SingleHiddenLayer'

environment = {
    'name': 'Pendulum-v0',
    'max_steps': 1000,
    'action_bins': (10,)
}

model = {
    'state_size': 3,
    'action_size': 9,
    'fc_units': 24
}

agent = {
    'lr': 0.001
}

train = {
    'n_episodes': 10000,
    'max_t': 1000,
    'beta': 0.0
}
