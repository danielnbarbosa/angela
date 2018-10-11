agent_type='dqn'
env_class='UnityMLVisual'
model_class='Conv3D2x'

environment = {
    'name': 'compiled_unity_environments/VisualBanana.app'
}

model = {
    'state_size': (3, 4, 84, 84),
    'action_size': 4,
}

agent = {
    'action_size': 4,
    'use_double_dqn': False,
    'buffer_size': 10000,
    'gamma': 0.99,
    'lr': 5e-4,
}

train = {
    'n_episodes': 10000,
    'solve_score': 13.0,
    'eps_start': 1.0,
    'eps_end': 0.01,
    'eps_decay': 0.995
}
