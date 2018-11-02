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
    'evaluation_only': True
}

train = {
    'n_episodes': 10000,
    'max_t': 350,
    'solve_score': 200.0
}
