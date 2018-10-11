agent_type='hc'
env_class='Gym'
model_class='SingleLayerPerceptron'

environment = {
    'name': 'LunarLander-v2',
    'seed': 888417152
}

model = {
    'state_size': 8,
    'action_size': 4,
    'seed': 888417152
}

agent = {
    'action_size': 4,
    'policy': 'stochastic',
    'seed': 888417152
}

train = {
    'n_episodes': 1500,
    'max_t': 2000,
    'solve_score': 200.0,
    'seed': 888417152,
    'npop': 3
}
