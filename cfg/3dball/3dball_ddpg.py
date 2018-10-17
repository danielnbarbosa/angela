agent_type='ddpg'
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
    'update_every': 2,
    'n_agents': 12,
    'batch_size': 128,
    'weight_decay': 0.0,
    #'evaluation_only': True
}

train = {
    'n_episodes': 100000,
    'solve_score': 100.0,
}
