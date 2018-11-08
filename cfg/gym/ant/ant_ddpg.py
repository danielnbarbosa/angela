algorithm='ddpg'
env_class='Gym'
model_class='LowDim2x'

environment = {
    'name': 'Ant-v2',
}

model = {
    'state_size': 111,
    'action_size': 8,
    'fc1_units': 400,
    'fc2_units': 300
}

agent = {
    'action_size': 8,
    #'update_every': 2,
    #'buffer_size': int(3e5),
    #'batch_size': 128,
    #'weight_decay': 0.0,
    'sigma': 0.1
}

train = {
    'n_episodes': 100000,
    'max_t': 200,
    'solve_score': 6000.0,
}
