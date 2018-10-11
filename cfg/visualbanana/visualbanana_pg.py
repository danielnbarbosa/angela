agent_type='pg'
env_class='UnityMLVisual'
model_class='Conv3D'

environment = {
    'name': 'compiled_unity_environments/VisualBanana.app'
}

model = {
    'state_size': (3, 4, 84, 84),
    'action_size': 4
}

agent = {
    'lr': 0.0001
}

train = {
    'n_episodes': 10000,
    'solve_score': 13.0
}
