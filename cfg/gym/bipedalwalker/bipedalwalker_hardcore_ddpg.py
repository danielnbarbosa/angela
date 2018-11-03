algorithm='ddpg'
env_class='Gym'
model_class='LowDim2x'

environment = {
    'name': 'BipedalWalkerHardcore-v2'
}

model = {
    'state_size': 24,
    'action_size': 4
}

agent = {
    'action_size': 4,
    'update_every': 2,
    'batch_size': 128,
    'gamma': 0.98
}

train = {
    'n_episodes': 100000,
    'max_t': 2000,
    'solve_score': 300.0
}
