from libs import environments, models, agents, train

SEED = 0
#SEED = random.randint(0, 2 ** 30)

environment = environments.Gym('FrozenLake8x8-v0', seed=SEED, one_hot=64)


def dqn(render, load_file):
    model = models.dqn.Dueling2x(state_size=64, action_size=4, fc_units=(64, 64), seed=SEED)
    agent = agents.DQN(model, action_size=4, seed=SEED,
                  use_double_dqn=True,
                  use_prioritized_experience_replay=False)
    train.dqn(environment, agent, n_episodes=6000, max_t=1000)
