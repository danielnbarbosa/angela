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
    'fc2_units': 64
}

agent = {
    'action_size': 2,
    'evaluation_only': True
}

train = {
    'n_episodes': 100000,
    'max_t': 1000,
    #'solve_score': 2000.0,
}
