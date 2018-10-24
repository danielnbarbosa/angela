algorithm='ddpg'
env_class='UnityMLVectorMultiAgentNew'
model_class='LowDim2x'

environment = {
    'name': 'compiled_unity_environments/Crawler_v5.app',
}

model = {
    'state_size': 129,
    'action_size': 20,
    'fc1_units': 400,
    'fc2_units': 300
}

agent = {
    'action_size': 20,
    'n_agents': 12,
    'evaluation_only': True
}

train = {
    'n_episodes': 100000,
    'max_t': 1000,
    'solve_score': 2000.0,
}
