agent_type='hc'
env_class='Gym'
model_class='SingleLayerPerceptron'

environment = {
    'name': 'Acrobot-v1',
    'max_steps': 1000
}

model = {
    'state_size': 6,
    'action_size': 3
}

agent = {
    'action_size': 3,
    'policy': 'deterministic'
}

train = {
    'n_episodes': 1000,
    'max_t': 1000,
    'npop': 5
}
