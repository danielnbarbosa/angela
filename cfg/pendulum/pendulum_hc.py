agent_type='hc'
env_class='Gym'
model_class='SingleLayerPerceptron'

environment = {
    'name': 'Pendulum-v0',
    'max_steps': 1000,
    'action_bins': (10,)
}

model = {
    'state_size': 3,
    'action_size': 9
}

agent = {
    'action_size': 9,
    'policy': 'stochastic'
}

train = {
    'n_episodes': 2000,
    'max_t': 1000,
    'npop': 10
}
