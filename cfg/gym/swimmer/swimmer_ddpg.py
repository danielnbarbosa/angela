algorithm='ddpg'
env_class='Gym'
model_class='LowDim2x'

environment = {
    'name': 'Swimmer-v2',
}

model = {
    'state_size': 8,
    'action_size': 2,
    'fc1_units': 128,
    'fc2_units': 64,
    #'seed': 42
}

agent = {
    'action_size': 2,
    'update_every': 2,
    'buffer_size': int(2e5),
    'batch_size': 64,
    'sigma': 0.3,
    #'weight_decay': 0.0,
}

train = {
    'n_episodes': 100000,
    'max_t': 300,
}
