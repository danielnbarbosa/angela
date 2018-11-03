algorithm='dqn'
env_class='Gym'
model_class='Dueling2x'

environment = {
    'name': 'MountainCar-v0',
    'max_steps': 1000
}

model = {
    'state_size': 2,
    'action_size': 3,
    'fc_units': (64, 64)
}

agent = {
    'action_size': 3,
    'use_double_dqn': True
}

train = {
    'n_episodes': 4000,
    'max_t': 1000,
    'solve_score': -110.0,
    'eps_start': 1.0,
    'eps_end': 0.05,
    'eps_decay': 0.997
}
