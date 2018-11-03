algorithm='pg'
env_class='Gym'
model_class='SingleHiddenLayer'

environment = {
    'name': 'MountainCar-v0',
    'max_steps': 1000
}

model = {
    'state_size': 2,
    'action_size': 3,
    'fc_units': 16
}

agent = {
    'lr': 0.005
}

train = {
    'n_episodes': 5000,
    'max_t': 1000,
    'solve_score': -110.0
}
