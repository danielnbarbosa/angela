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
    'fc2_units': 300
}

agent = {
    'action_size': 17,
    'evaluation_only': True
}

train = {
    'n_episodes': 100000,
    'solve_score': 3800.0,
}
