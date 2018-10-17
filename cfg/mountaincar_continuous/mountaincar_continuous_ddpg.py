algorithm='ddpg'
env_class='Gym'
model_class='LowDim2x'

environment = {
    'name': 'MountainCarContinuous-v0'
}

model = {
    'state_size': 2,
    'action_size': 1
}

agent = {
    'action_size': 1,
    'update_every': 1,
    'weight_decay': 0.0001
}

train = {
    'n_episodes': 10000,
    'max_t': 1000,
    'solve_score': 90.0
}
