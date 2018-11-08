algorithm='ddpg'
env_class='Gym'
model_class='LowDim2x'

environment = {
    'name': 'HalfCheetah-v2',
}

model = {
    'state_size': 17,
    'action_size': 6,
    'fc1_units': 128,
    'fc2_units': 64
}

agent = {
    'action_size': 2,
    'evaluation_only': True
}

train = {
    'n_episodes': 100000,
    'solve_score': 4800.0,
}
