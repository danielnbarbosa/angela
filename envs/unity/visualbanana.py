from libs import environments, models, agents, training

"""
NOTE: Download pre-built Unity VisualBannana.app from: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/
"""

SEED = 0
#SEED = random.randint(0, 2 ** 30)

environment = environments.UnityMLVisual('envs/unity/compiled_unity_environments/VisualBanana.app', seed=SEED)
#environment = environments.UnityMLVisual('envs/unity/compiled_unity_environments/VisualBanana_Linux/Banana.x86_64', seed=SEED)

def dqn(render, load_file):
    # shape is (m, c, f, h, w)
    model = models.dqn.Conv3D2x(state_size=(3, 4, 84, 84), action_size=4, seed=SEED)
    agent = agents.DQN(model, action_size=4, seed=SEED,
                     buffer_size=10000,
                     gamma=0.99,
                     lr=5e-4,
                     use_double_dqn=False,
                     use_prioritized_experience_replay=False)
    # don't forget to reset epsilon when continuing training
    training.train_dqn(environment, agent, n_episodes=10000,
              solve_score=13.0,
              eps_start=1.0,
              eps_end=0.01,
              eps_decay=0.995)


def pg(render, load_file):
    model = models.pg.Conv3D(state_size=(3, 4, 84, 84), action_size=4, seed=SEED)
    agent = agents.PolicyGradient(model, seed=SEED,
                                  lr=0.0001,
                                  load_file=load_file)
    training.train_pg(environment, agent, n_episodes=10000,
             solve_score=13.0,
             gamma=0.99,
             render=render,
             graph_when_done=False)
