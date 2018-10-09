agent_type='dqn'
env_class='UnityMLVector'
model_class='TwoLayer2x'

environment = {
    'name': 'cfg/compiled_unity_environments/Basic.app'
}

model = {
    'state_size': 1,
    'action_size': 2,
    'fc_units': (8, 8)
}

agent = {
    'action_size': 2,
    'use_double_dqn': True,
    'buffer_size': 5000
}

train = {
    'n_episodes': 4000,
    'solve_score': 0.94,
    'eps_start': 1.0,
    'eps_end': 0.01,
    'eps_decay': 0.999
}
