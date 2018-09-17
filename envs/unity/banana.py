from imports import *

"""
NOTE: download pre-built Unity Bannana.app from: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/
"""

SEED = 0
#SEED = random.randint(0, 2 ** 30)

environment = environments.UnityMLVector('envs/unity/compiled_unity_environments/Banana.app', seed=SEED)


def dqn():
    #SEED=895815691
    model = models.DQNTwoHiddenLayer_Q(state_size=37, action_size=4, fc_units=(32, 32), seed=SEED)
    agent = agents.DQN(model, action_size=4, seed=SEED,
                     use_double_dqn=False,
                     use_prioritized_experience_replay=False)
    train_dqn(environment, agent, n_episodes=1000, solve_score=13.0,
              eps_start=1.0,
              eps_end=0.001,
              eps_decay=0.97)


def hc():
    agent = agents.HillClimbing(state_size=37, action_size=4, seed=SEED, policy='stochastic')
    train_hc(environment, agent, seed=SEED, n_episodes=2000, solve_score=13.0,
             use_adaptive_noise=True,
             npop=6,
             print_every=20,
             graph_when_done=True)
