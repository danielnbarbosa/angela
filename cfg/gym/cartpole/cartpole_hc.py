algorithm='hc'
env_class='Gym'
model_class='SingleLayerPerceptron'

environment = {
    'name': 'CartPole-v1',
    'max_steps': 1000
}

model = {
    'state_size': 4,
    'action_size': 2
}

agent = {
    'action_size': 2,
    'use_adaptive_noise': False,
    'policy': 'deterministic'
}

train = {
    'n_episodes': 4000,
    'max_t': 1000,
    'solve_score': 195.0,
    'npop': 10
}
