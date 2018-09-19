from libs import environments, models, agents, train

SEED = 0
#SEED = random.randint(0, 2 ** 30)

environment = environments.Gym('MountainCar-v0', seed=SEED, max_steps=1000)


def dqn(render, load_file):
    model = models.dqn.Dueling2x(state_size=2, action_size=3, fc_units=(64, 64), seed=SEED)
    agent = agents.DQN(model, action_size=3, seed=SEED,
                     use_double_dqn=True,
                     use_prioritized_experience_replay=False)
    train.dqn(environment, agent, n_episodes=4000, max_t=1000, solve_score=-110.0,
              eps_start=1.0,
              eps_end=0.05,
              eps_decay=0.997)


def hc(render, load_file):
    model = models.hc.SingleLayerPerceptron(state_size=2, action_size=3, seed=SEED)
    agent = agents.HillClimbing(model, action_size=3, seed=SEED, policy='stochastic')
    train.hc(environment, agent, seed=SEED, n_episodes=4000, max_t=1000,
             npop=4,
             solve_score=-110.0,
             graph_when_done=False)


def pg(render, load_file):
    model = models.pg.SingleHiddenLayer(state_size=2, action_size=3, fc1_units=16, seed=SEED)
    agent = agents.PolicyGradient(model, seed=SEED, lr=0.005)
    train.pg(environment, agent, n_episodes=5000, max_t=1000,
             gamma=0.99,
             solve_score=-110.0,
             graph_when_done=False)
