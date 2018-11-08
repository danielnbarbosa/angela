algorithm='ddpg'
env_class='Gym'
model_class='LowDim2x'

environment = {
    'name': 'Ant-v2',
}

model = {
    'state_size': 111,
    'action_size': 8,
    'fc1_units': 400,
    'fc2_units': 300
}

agent = {
    'action_size': 8,
    'evaluation_only': True
}

train = {
    'n_episodes': 100000,
    'max_t': 1000,
    'solve_score': 6000.0,
}
