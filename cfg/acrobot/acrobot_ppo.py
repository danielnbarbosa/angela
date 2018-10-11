agent_type='ppo'
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
    'lr': 0.001
}

train = {
    'n_episodes':   4000,
    'max_t':        1000,
    'epsilon':      0.1,
    'beta':         0.0,
    'sgd_epoch':    4,
    'sample_epoch': 5
}
