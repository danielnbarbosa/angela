algorithm='hc'
env_class='UnityMLVector'
model_class='SingleLayerPerceptron'

environment = {
    'name': 'compiled_unity_environments/Banana.app'
}

model = {
    'state_size': 37,
    'action_size': 4
}

agent = {
    'action_size': 4,
    'policy': 'stochastic'
}

train = {
    'n_episodes': 2000,
    'solve_score': 13.0,
    'npop': 6
}
