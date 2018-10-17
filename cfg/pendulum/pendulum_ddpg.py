algorithm='ddpg'
env_class='Gym'
model_class='LowDim2x'

environment = {
    'name': 'Pendulum-v0',
    'max_steps': 300,
    'seed': 2
}

model = {
    'state_size': 3,
    'action_size': 1,
    'seed': 2
}

agent = {
    'action_size': 1,
    'update_every': 4,
    'weight_decay': 0.0,
    'seed': 2
}

train = {
    'n_episodes': 4000,
    'max_t': 300
}
