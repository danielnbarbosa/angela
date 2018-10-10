agent_type='ddpg'
env_class='UnityMLVectorMultiAgent'
model_class='LowDim2x'

environment = {
    'name': 'cfg/compiled_unity_environments/Crawler_Linux/Crawler.x86_64',
}

model = {
    'state_size': 129,
    'action_size': 20,
}

agent = {
    'action_size': 20,
    'update_every': 2,
    'n_agents': 12,
    'batch_size': 128,
}

train = {
    'n_episodes': 1000000,
    'solve_score': 2000.0,
}
