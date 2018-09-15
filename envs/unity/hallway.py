from imports import *

"""
NOTE: Compile and build the Basic.app with Unity using the scene in ml-agents.
      Need to make some modifications to support multiple simultaneous actions.
"""

SEED = 0
#SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = environments.UnityMLVector('env/unity/compiled_unity_environments/Hallway.app', seed=SEED)


def dqn():
    model = models.DQNTwoHiddenLayer_Q(state_size=36, action_size=16, fc_units=(64, 64), seed=SEED)
    agent = agents.DQN(model, action_size=16, seed=SEED,
                     use_double_dqn=False,
                     use_prioritized_experience_replay=False)
    train_dqn(environment, agent, n_episodes=1000, solve_score=0.7,
              eps_start=1,
              eps_end=0.001,
              eps_decay=0.97)
