"""
NOTE: Download pre-built Unity Bannana.app from: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher
"""

agent_type='ddpg'
env_class='UnityMLVector'
model_class='LowDim2x'

environment = {
    'name': 'cfg/compiled_unity_environments/Reacher.app'
}

model = {
    'state_size': 33,
    'action_size': 4,
}

agent = {
    'action_size': 4,
    'update_every': 2,
    'buffer_size': int(1e5),
    'clip_critic_gradients': True
}

train = {
    'n_episodes': 2000,
    'solve_score': 30.0,
    'graph_when_done': True
}
