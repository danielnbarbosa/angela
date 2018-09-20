from libs import environments, models, agents, train

SEED = 0
#SEED = random.randint(0, 2 ** 30)

environment = environments.Gym('Acrobot-v1', seed=SEED)


def dqn(render, load_file):
    model = models.dqn.Dueling2x(state_size=6, action_size=3, fc_units=(64, 64), seed=SEED)
    agent = agents.DQN(model, action_size=3, seed=SEED, load_file=load_file,
                     use_double_dqn=True,
                     use_prioritized_experience_replay=False)
    train.dqn(environment, agent, n_episodes=12000, max_t=1000)


def hc(render, load_file):
    model = models.hc.SingleLayerPerceptron(state_size=6, action_size=3, seed=SEED)
    agent = agents.HillClimbing(model, action_size=3, seed=SEED, load_file=load_file,
                                policy='deterministic')
    train.hc(environment, agent, seed=SEED, n_episodes=1000, max_t=1000,
             npop=5,
             graph_when_done=False)


def pg(render, load_file):
    model = models.pg.SingleHiddenLayer(state_size=6, action_size=3, fc1_units=32, seed=SEED)
    agent = agents.PolicyGradient(model, seed=SEED, load_file=load_file, lr=0.005)
    train.pg(environment, agent, n_episodes=4000, max_t=1000,
             gamma=0.99,
             graph_when_done=False)
