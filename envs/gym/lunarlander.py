from imports import *

SEED = 0
#SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = environments.Gym('LunarLander-v2', seed=SEED)


def dqn():
    model = models.DQNDueling_Q(state_size=8, action_size=4, fc_units=(128, 128), seed=SEED)
    agent = agents.DQN(model, action_size=4, seed=SEED,
                    use_double_dqn=True,
                    use_prioritized_experience_replay=False,
                    buffer_size=100000)
    train_dqn(environment, agent, n_episodes=4000, max_t=2000, solve_score=200.0)


def hc():
    agent = agents.HillClimbing(state_size=8, action_size=4, seed=SEED, policy='deterministic')
    train_hc(environment, agent, seed=SEED, n_episodes=1500, max_t=2000,
             use_adaptive_noise=True,
             npop=10,
             print_every=100,
             solve_score=200.0,
             graph_when_done=True)


def pg():
    model = models.PGOneHiddenLayer(state_size=8, action_size=4, fc1_units=32, seed=SEED)
    agent = agents.PolicyGradient(model, seed=SEED, lr=0.005)
    train_pg(environment, agent, seed=SEED, n_episodes=5000, max_t=2000,
             gamma=0.99,
             graph_when_done=True)
