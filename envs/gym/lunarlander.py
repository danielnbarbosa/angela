from imports import *

SEED = 0
#SEED = random.randint(0, 2 ** 30)

environment = environments.Gym('LunarLander-v2', seed=SEED)


def dqn(render):
    model = models.DQNDueling_Q(state_size=8, action_size=4, fc_units=(128, 128), seed=SEED)
    agent = agents.DQN(model, action_size=4, seed=SEED,
                    use_double_dqn=True,
                    use_prioritized_experience_replay=False,
                    buffer_size=100000)
    train_dqn(environment, agent, n_episodes=4000, max_t=2000, solve_score=200.0)


def hc(render):
    model = models.HillClimbing(state_size=8, action_size=4, seed=SEED)
    agent = agents.HillClimbing(model, action_size=4, seed=SEED, policy='deterministic')
    train_hc(environment, agent, seed=SEED, n_episodes=1500, max_t=2000,
             solve_score=200.0,
             graph_when_done=True)


def pg(render):
    model = models.PGOneHiddenLayer(state_size=8, action_size=4, fc1_units=32, seed=SEED)
    agent = agents.PolicyGradient(model, seed=SEED, lr=0.005)
    train_pg(environment, agent, n_episodes=5000, max_t=2000,
             gamma=0.99,
             graph_when_done=True)
