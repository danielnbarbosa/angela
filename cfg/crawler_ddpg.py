agent_type='ddpg'
env_class='UnityMLVectorMultiAgent'
model_class='LowDim2x'

environment = {
    'name': 'cfg/compiled_unity_environments/Crawler.app',
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
    'n_episodes': 10000,
    'solve_score': 2000.0,
    'graph_when_done': True
}
