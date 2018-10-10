agent_type='ddpg'
env_class='UnityMLVector'
model_class='LowDim2x'

environment = {
    'name': 'cfg/compiled_unity_environments/Reacher.app'
}

model = {
    'state_size': 33,
    'action_size': 4,
}

agent = {
    'action_size': 4,
    'update_every': 2,
}

train = {
    'n_episodes': 2000,
    'solve_score': 30.0,
    'graph_when_done': True
}
