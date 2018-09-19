from libs import environments, models, agents, train

SEED = 0
#SEED = random.randint(0, 2 ** 30)

environment = environments.PLEFlappyBird(seed=SEED)

def dqn(render, load_file):
    model = models.dqn.SmallConv2D2x(state_size=(4, 80, 80), action_size=2, fc_units=256, seed=SEED, normalize=True)
    agent = agents.DQN(model, action_size=2, seed=SEED,
                     use_double_dqn=False,
                     use_prioritized_experience_replay=False)
    train.dqn(environment, agent, n_episodes=50000, max_t=1000,
              eps_start=1.0,
              eps_end=0.0001,
              eps_decay=0.9999)


def pg(render, load_file):
    model = models.pg.SmallConv2D(state_size=(4, 80, 80), action_size=2, fc_units=256, seed=SEED, normalize=True)
    agent = agents.PolicyGradient(model, seed=SEED,
                                  lr=0.0001,
                                  load_file=load_file,
                                  action_map={0: None, 1: 119})
    train.pg(environment, agent, n_episodes=50000, max_t=2000,
             gamma=0.99,
             render=render,
             graph_when_done=False)
