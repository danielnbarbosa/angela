algorithm='hc'
env_class='Gym'
model_class='SingleLayerPerceptron'

environment = {
    'name': 'MountainCar-v0',
    'max_steps': 1000
}

model = {
    'state_size': 2,
    'action_size': 3
}

agent = {
    'action_size': 3,
    'policy': 'stochastic'
}

train = {
    'n_episodes': 4000,
    'max_t': 1000,
    'solve_score': -110.0,
    'npop': 4
}
