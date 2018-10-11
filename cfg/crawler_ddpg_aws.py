agent_type='ddpg'
env_class='UnityMLVectorMultiAgent'
model_class='LowDim2x'

environment = {
    'name': 'cfg/compiled_unity_environments/Crawler_Linux/Crawler.x86_64',
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
    'batch_size': 256,
    'buffer_size': int(2e5),
}

train = {
    'n_episodes': 100000,
    'solve_score': 2000.0,
}
