"""
NOTE: Download pre-built Unity Bannana.app from: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/
"""


agent_type='hc'
env_class='UnityMLVector'
model_class='SingleLayerPerceptron'

environment = {
    'name': 'cfg/compiled_unity_environments/Banana.app'
}

model = {
    'state_size': 37,
    'action_size': 4
}

agent = {
    'action_size': 4,
    'policy': 'stochastic'
}

train = {
    'n_episodes': 2000,
    'solve_score': 13.0,
    'npop': 6
}
