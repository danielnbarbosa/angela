algorithm='ddpg'
env_class='UnityMLVector'
model_class='LowDim2x'

environment = {
    'name': 'compiled_unity_environments/Reacher_Linux/Reacher.x86_64'
}

model = {
    'state_size': 33,
    'action_size': 4,
}

agent = {
    'action_size': 4,
    'update_every': 2,
    'buffer_size': int(1e5),
}

train = {
    'n_episodes': 2000,
    'solve_score': 30.0,
}
