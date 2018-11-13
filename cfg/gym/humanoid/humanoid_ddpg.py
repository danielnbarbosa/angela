algorithm='ddpg'
env_class='Gym'
model_class='LowDim2x'

environment = {
    'name': 'Humanoid-v2',
}

model = {
    'state_size': 376,
    'action_size': 17,
    'fc1_units': 400,
    'fc2_units': 300,
}

agent = {
    'action_size': 17,
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
