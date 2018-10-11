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
    'update_every': 4,
    'n_agents': 12,
    'batch_size': 128,
    'buffer_size': int(1e5),
    'clip_critic_gradients': True
}

train = {
    'n_episodes': 100000,
    'solve_score': 2000.0,
    'graph_when_done': True
}
