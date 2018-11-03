algorithm='ddpg'
env_class='Gym'
model_class='LowDim2x'

environment = {
    'name': 'LunarLanderContinuous-v2'
}

model = {
    'state_size': 8,
    'action_size': 2
}

agent = {
    'action_size': 2,
    'update_every': 2,
    'batch_size': 128,
    'weight_decay': 0.0001
}

train = {
    'n_episodes': 10000,
    'max_t': 350,
    'solve_score': 200.0
}
