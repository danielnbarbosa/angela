from libs import environments, models, agents, train

SEED = 0
#SEED = random.randint(0, 2 ** 30)

environment = environments.Gym('Carnival-ram-v0', seed=SEED, normalize=True)


def dqn(render, load_file):
    model = models.dqn.Dueling2x(state_size=128, action_size=6, fc_units=(256, 128), seed=SEED)
    agent = agents.DQN(model, action_size=6, seed=SEED,
              use_double_dqn=True,
              use_prioritized_experience_replay=False,
              buffer_size=100000)
    train.dqn(environment, agent, n_episodes=30000, max_t=10000,
              eps_start=1,
              eps_end=0.1,
              eps_decay=0.9999,
              graph_when_done=True)


def hc(render, load_file):
    model = models.hc.SingleLayerPerceptron(state_size=128, action_size=6, seed=SEED)
    agent = agents.HillClimbing(model, action_size=6, seed=SEED, policy='deterministic')
    train.hc(environment, agent, seed=SEED, n_episodes=4000, max_t=2000,
             npop=10,
             graph_when_done=False)