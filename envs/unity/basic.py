from imports import *

"""
NOTE: Compile and build the Basic.app with Unity using the scene in ml-agents.
"""

SEED = 0
#SEED = random.randint(0, 2 ** 30)

environment = environments.UnityMLVector('envs/unity/compiled_unity_environments/Basic.app', seed=SEED)


def dqn():
    model = models.DQNTwoHiddenLayer_Q(state_size=1, action_size=2, fc_units=(8, 8), seed=SEED)
    agent = agents.DQN(model, action_size=2, seed=SEED,
                     use_double_dqn=False,
                     use_prioritized_experience_replay=False)
    train_dqn(environment, agent, n_episodes=1000, solve_score=0.93,
              eps_start=1,
              eps_end=0.001,
              eps_decay=0.97)
