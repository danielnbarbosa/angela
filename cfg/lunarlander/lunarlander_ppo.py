agent_type='ppo'
env_class='Gym'
model_class='SingleHiddenLayer'

environment = {
    'name': 'LunarLander-v2',
}

model = {
    'state_size': 8,
    'action_size': 4,
    'fc_units': 32
}

agent = {
    'lr': 0.005
}

train = {
    'n_episodes':   5000,
    'max_t':        2000,
    'solve_score':  200.0,
    'epsilon':      0.3,
    'beta':         0.0,
    'sgd_epoch':    4,
    'n_trajectories': 1
}
