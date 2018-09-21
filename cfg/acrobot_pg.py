agent_type='pg'
env_class='Gym'
model_class='SingleHiddenLayer'

environment = {
    'name': 'Acrobot-v1',
}

model = {
    'state_size': 6,
    'action_size': 3,
    'fc_units': 32
}

agent = {
    'lr': 0.005
}

train = {
    'n_episodes': 4000,
    'max_t': 1000
}
