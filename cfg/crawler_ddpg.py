agent_type='ddpg'
env_class='UnityMLVectorMultiAgent'
model_class='LowDim2x'

environment = {
    'name': 'cfg/compiled_unity_environments/Crawler.app',
}

model = {
    'state_size': 129,
    'action_size': 20,
    'fc1_units': 400,
    'fc2_units': 300
}

agent = {
    'action_size': 20,
    'update_every': 1,
    'n_agents': 12,
    'batch_size': 64,
    'buffer_size': int(1e5),
    'inference_only': True
}

train = {
    'n_episodes': 100000,
    'max_t': 1000,
    'solve_score': 2000.0,
}
