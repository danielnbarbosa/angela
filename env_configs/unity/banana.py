from libs import environments, models, agents, train

"""
NOTE: Download pre-built Unity Bannana.app from: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/
"""

SEED = 0
#SEED = random.randint(0, 2 ** 30)

environment = environments.UnityMLVector('env_configs/unity/compiled_unity_environments/Banana.app', seed=SEED)


def dqn(render, load_file):
    #SEED=895815691
    model = models.dqn.TwoLayer2x(state_size=37, action_size=4, fc_units=(32, 32), seed=SEED)
    agent = agents.DQN(model, action_size=4, seed=SEED,
                     use_double_dqn=False,
                     use_prioritized_experience_replay=False)
    train.dqn(environment, agent, n_episodes=1000, solve_score=13.0,
              eps_start=1.0,
              eps_end=0.001,
              eps_decay=0.97)


def hc(render, load_file):
    model = models.hc.SingleLayerPerceptron(state_size=37, action_size=4, seed=SEED)
    agent = agents.HillClimbing(model, action_size=4, seed=SEED, policy='stochastic')
    train.hc(environment, agent, seed=SEED, n_episodes=2000, solve_score=13.0,
             npop=6,
             graph_when_done=False)
