agent_type='dqn'
env_class='Gym'
model_class='Dueling2x'

environment = {
    'name': 'LunarLander-v2',
    'seed': 42
}

model = {
    'state_size': 8,
    'action_size': 4,
    'fc_units': (128, 128),
    'seed': 42
}

agent = {
    'action_size': 4,
    'use_double_dqn': True,
    'buffer_size': 100000,
    'seed': 42
}

train = {
    'n_episodes': 4000,
    'max_t': 2000,
    'solve_score': 200.0
}
