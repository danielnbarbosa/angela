algorithm='ddpg'
env_class='UnityMLVectorMultiAgentNew'
model_class='LowDim2x'

environment = {
    'name': 'compiled_unity_environments/Crawler_Linux_v5/CrawlerLinux.x86_64',
}

model = {
    'state_size': 129,
    'action_size': 20,
    'fc1_units': 400,
    'fc2_units': 300
}

agent = {
    'action_size': 20,
    'update_every': 2,
    'n_agents': 12,
    'batch_size': 64,
    'buffer_size': int(3e5),
    'clip_gradients': True
}

train = {
    'n_episodes': 1000000,
    'max_t': 1000,
    'solve_score': 2000.0,
}
