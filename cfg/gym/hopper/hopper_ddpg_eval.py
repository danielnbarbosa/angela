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
    'fc2_units': 64
}

agent = {
    'action_size': 3,
    'evaluation_only': True
}

train = {
    'n_episodes': 100000,
    'solve_score': 3800.0,
}
