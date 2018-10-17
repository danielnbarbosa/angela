algorithm='ppo'
env_class='UnityMLVectorMultiAgentNew'
model_class='SingleHiddenLayer'

environment = {
    'name': 'compiled_unity_environments/Hallway.app'
}

model = {
    'state_size': 36,
    'action_size': 5,
    'fc_units': 64
}

agent = {
    'lr': 0.001,
    'n_agents': 16,
}

train = {
    'n_episodes':     4000,
    'max_t':          1000,
    'epsilon':        0.1,
    'sgd_epoch':      4,
    'n_trajectories': 1,
    'solve_score':    0.7,
}
