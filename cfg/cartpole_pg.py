agent_type='pg'
env_class='Gym'
model_class='SingleHiddenLayer'

environment = {
    'name': 'CartPole-v1',
    'max_steps': 1000,
    'seed': 5
}

model = {
    'state_size': 4,
    'action_size': 2,
    'fc_units': 16
}

agent = {
    'lr': 0.005
}

train = {
    'n_episodes': 4000,
    'max_t': 1000,
    'solve_score': 195.0
}
