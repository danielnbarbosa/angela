algorithm='maddpg_v1'
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
    #'evaluation_only': True
}

train = {
    'n_episodes': 100000,
    'solve_score': 0.5,
    'graph_when_done': True
}
