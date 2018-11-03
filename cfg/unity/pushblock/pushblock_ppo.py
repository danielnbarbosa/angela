algorithm='ppo'
env_class='UnityMLVectorMultiAgentNew'
model_class='SingleHiddenLayer'

environment = {
    'name': 'compiled_unity_environments/PushBlock.app'
}

model = {
    'state_size': 210,
    'action_size': 7,
    'fc_units': 64
}

agent = {
    'lr': 0.0005,
    'n_agents': 32,
}

train = {
    'n_episodes':     4000,
    'max_t':          200,
    'epsilon':        0.1,
    'beta':           0.0,
    'sgd_epoch':      4,
    'n_trajectories': 1
}
