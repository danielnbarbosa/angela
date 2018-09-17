from imports import *

SEED = 0
#SEED = random.randint(0, 2 ** 30)

environment = environments.Gym('Acrobot-v1', seed=SEED)


def dqn(render):
    model = models.DQNDueling_Q(state_size=6, action_size=3, fc_units=(64, 64), seed=SEED)
    agent = agents.DQN(model, action_size=3, seed=SEED,
                     use_double_dqn=True,
                     use_prioritized_experience_replay=False)
    train_dqn(environment, agent, n_episodes=12000, max_t=1000)


def hc(render):
    model = models.HillClimbing(state_size=6, action_size=3, seed=SEED)
    agent = agents.HillClimbing(model, action_size=3, seed=SEED, policy='deterministic')
    train_hc(environment, agent, seed=SEED, n_episodes=1000, max_t=1000,
             npop=5,
             graph_when_done=True)


def pg(render):
    model = models.PGOneHiddenLayer(state_size=6, action_size=3, fc1_units=32, seed=SEED)
    agent = agents.PolicyGradient(model, seed=SEED, lr=0.005)
    train_pg(environment, agent, n_episodes=4000, max_t=1000,
             gamma=0.99,
             graph_when_done=True)
