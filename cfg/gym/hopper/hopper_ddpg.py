algorithm='ddpg'
env_class='Gym'
model_class='LowDim2x'

environment = {
    'name': 'Hopper-v2',
}

model = {
    'state_size': 11,
    'action_size': 3,
    'fc1_units': 128,
    'fc2_units': 64,
}

agent = {
    'action_size': 3,
    'update_every': 2,
    'buffer_size': int(1e5),
    'batch_size': 64,
    'sigma': 0.2,
    #'weight_decay': 0.01,
}

train = {
    'n_episodes': 100000,
    'max_t': 1000,
    'solve_score': 3800.0,
}
