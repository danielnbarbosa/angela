from libs import environments, models, agents, train

SEED = 0
#SEED = random.randint(0, 2 ** 30)

environment = environments.Gym('Pendulum-v0', seed=SEED, max_steps=1000, action_bins=(10,))


def dqn(render, load_file):
    model = models.dqn.Dueling2x(state_size=3, action_size=9, fc_units=(32, 32), seed=SEED)
    agent = agents.DQN(model, action_size=9, seed=SEED,
                  use_double_dqn=True,
                  use_prioritized_experience_replay=False)
    train.dqn(environment, agent, n_episodes=4000, max_t=1000)


def hc(render, load_file):
    model = models.hc.SingleLayerPerceptron(state_size=3, action_size=9, seed=SEED)
    agent = agents.HillClimbing(model, action_size=9, seed=SEED, policy='stochastic')
    train.hc(environment, agent, seed=SEED, n_episodes=2000, max_t=1000,
             npop=10,
             graph_when_done=False)


def pg(render, load_file):
    model = models.pg.SingleHiddenLayer(state_size=3, action_size=9, fc1_units=24, seed=SEED)
    agent = agents.PolicyGradient(model, seed=SEED, lr=0.005)
    train.pg(environment, agent, n_episodes=10000, max_t=1000,
             gamma=0.99,
             graph_when_done=False)
