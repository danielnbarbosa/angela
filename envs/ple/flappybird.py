from libs import environments, models, agents, train

SEED = 0
#SEED = random.randint(0, 2 ** 30)

environment = environments.PLEFlappyBird(seed=SEED)


def pg(render, load_file):
    model = models.pg.SmallConv2D(state_size=(4, 80, 80), action_size=2, fc_units=200, seed=SEED)
    agent = agents.PolicyGradient(model, seed=SEED,
                                  lr=0.0001,
                                  load_file=load_file,
                                  action_map={0: None, 1: 119})
    train.pg(environment, agent, n_episodes=50000, max_t=10000,
             gamma=0.99,
             render=render,
             graph_when_done=False)