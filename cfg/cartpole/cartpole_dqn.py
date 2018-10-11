agent_type='dqn'
env_class='Gym'
model_class='Dueling2x'

environment = {
    'name': 'CartPole-v1',
    'max_steps': 1000
}

model = {
    'state_size': 4,
    'action_size': 2,
    'fc_units': (64, 32)
}

agent = {
    'action_size': 2,
    'gamma': 1.0,
    'use_double_dqn': True,
    'use_prioritized_experience_replay': False,
    'update_every': 4,
    'lr': 0.0006,
    'alpha_start': 0.5,
    'alpha_decay': 0.9992,
    'buffer_size': 100000
}

train = {
    'n_episodes': 1000,
    'max_t': 1000,
    'solve_score': 195.0,
    'eps_start': 1.0,
    'eps_end': 0.01,
    'eps_decay': 0.995
}
