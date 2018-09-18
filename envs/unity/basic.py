from libs import environments, models, agents, train

"""
NOTE: Compile and build the Basic.app with Unity using the scene in ml-agents.
"""

SEED = 0
#SEED = random.randint(0, 2 ** 30)

environment = environments.UnityMLVector('envs/unity/compiled_unity_environments/Basic.app', seed=SEED)


def dqn(render, load_file):
    model = models.dqn.TwoLayer2x(state_size=1, action_size=2, fc_units=(8, 8), seed=SEED)
    agent = agents.DQN(model, action_size=2, seed=SEED,
                     use_double_dqn=False,
                     use_prioritized_experience_replay=False,
                     buffer_size=5000)
    train.dqn(environment, agent, n_episodes=4000, solve_score=0.94,
              eps_start=1,
              eps_end=0.01,
              eps_decay=0.999)
