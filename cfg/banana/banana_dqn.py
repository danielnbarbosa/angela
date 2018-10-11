agent_type='dqn'
env_class='UnityMLVector'
model_class='TwoLayer2x'

environment = {
    'name': 'compiled_unity_environments/Banana.app'
}

model = {
    'state_size': 37,
    'action_size': 4,
    'fc_units': (32, 32)
}

agent = {
    'action_size': 4,
    'use_double_dqn': False,
}

train = {
    'n_episodes': 1000,
    'solve_score': 13.0,
    'eps_start': 1.0,
    'eps_end': 0.001,
    'eps_decay': 0.97
}
