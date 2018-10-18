algorithm='ddpg'
env_class='UnityMLVectorMultiAgent'
model_class='LowDim2x'

environment = {
    'name': 'compiled_unity_environments/Tennis.app',
}

model = {
    'state_size': 24,
    'action_size': 2,
}

agent = {
    'action_size': 2,
    'update_every': 2,
    'n_agents': 2,
    'buffer_size': int(1e5),
    'batch_size': 64,
    'gamma': 0.97
}

train = {
    'n_episodes': 100000,
    'solve_score': 0.5,
    'graph_when_done': True
}
