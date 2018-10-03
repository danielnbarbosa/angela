agent_type='ddpg'
env_class='Gym'
model_class='LowDim2x'

environment = {
    'name': 'BipedalWalker-v2'
}

model = {
    'state_size': 24,
    'action_size': 4
}

agent = {
    'action_size': 4,
    'update_every': 1,
    'buffer_size': 100000,
    'batch_size': 128,
    'weight_decay': 0.0001,
}

train = {
    'n_episodes': 4000,
    'max_t': 700,
    'solve_score': 300.0
}
