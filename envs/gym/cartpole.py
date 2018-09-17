from imports import *

SEED = 0
#SEED = random.randint(0, 2 ** 30)

environment = environments.Gym('CartPole-v1', seed=SEED, max_steps=1000)


def dqn(render, load_file):
    model = models.DQNDueling_Q(state_size=4, action_size=2, fc_units=(64, 32), seed=SEED)
    agent = agents.DQN(model, action_size=2, seed=SEED, load_file=load_file,
                     use_double_dqn=True,
                     use_prioritized_experience_replay=False,
                     update_every=4,
                     lr=0.0006,
                     alpha_start=0.5,
                     alpha_decay=0.9992,
                     buffer_size=100000)
    train_dqn(environment, agent, n_episodes=1000, max_t=1000,
              solve_score=195.0,
              eps_start=1.0,
              eps_end=0.01,
              eps_decay=0.995,
              render=render,
              graph_when_done=False)


def hc(render, load_file):
    model = models.HillClimbing(state_size=4, action_size=2, seed=SEED)
    agent = agents.HillClimbing(model, action_size=2, seed=SEED, load_file=load_file,
                                use_adaptive_noise=False,
                                policy='deterministic')
    train_hc(environment, agent, seed=SEED, n_episodes=4000, max_t=1000,
             npop=10,
             solve_score=195.0,
             render=render,
             graph_when_done=False)


def pg(render, load_file):
    model = models.PGOneHiddenLayer(state_size=4, action_size=2, fc1_units=16, seed=SEED)
    agent = agents.PolicyGradient(model, seed=SEED, load_file=load_file, lr=0.005)
    train_pg(environment, agent, n_episodes=4000, max_t=1000,
             solve_score=195.0,
             gamma=0.99,
             render=render,
             graph_when_done=False)
