algorithm='ddpg'
env_class='Gym'
model_class='LowDim2x'

environment = {
    'name': 'InvertedPendulum-v2',
}

model = {
    'state_size': 4,
    'action_size': 1,
    'fc1_units': 64,
    'fc2_units': 32
}

agent = {
    'action_size': 1,
    'update_every': 4,
    'buffer_size': int(1e5),
    'batch_size': 64,
}

train = {
    'n_episodes': 100000,
    'max_t': 1000,
}
