algorithm='ddpg'
env_class='UnityMLVectorMultiAgentNew'
model_class='LowDim2x'

environment = {
    'name': 'compiled_unity_environments/3DBall.app',
}

model = {
    'state_size': 8,
    'action_size': 2,
}

agent = {
    'action_size': 2,
    'n_agents': 12,
    'evaluation_only': True
}

train = {
    'n_episodes': 100000,
    'solve_score': 100.0,
}
