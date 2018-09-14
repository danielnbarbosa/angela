from imports import *

SEED = 0
#SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = GymEnvironment('CartPole-v1', seed=SEED, max_steps=1000)


def dqn():
    model = DuelingQNet(state_size=4, action_size=2, fc1_units=64, fc2_units=32, seed=SEED)
    agent = DQNAgent(model, action_size=2, seed=SEED,
                     use_double_dqn=True,
                     use_prioritized_experience_replay=True,
                     update_every=2,
                     lr=0.0006,
                     alpha_start=0.5,
                     alpha_decay=0.9992,
                     buffer_size=10000)
    train_dqn(environment, agent, n_episodes=1000, max_t=1000, solve_score=195.0,
              eps_start=1.0,
              eps_end=0.01,
              eps_decay=0.995)


def hc():
    agent = HillClimbingAgent(state_size=4, action_size=2, seed=SEED, policy='deterministic')
    train_hc(environment, agent, seed=SEED, n_episodes=4000, max_t=1000,
             use_adaptive_noise=False,
             npop=10,
             print_every=10,
             solve_score=195.0,
             graph_when_done=False)


def pg():
    model = SingleHiddenLayerWithSoftmaxOutput(state_size=4, action_size=2, fc1_units=16, seed=SEED)
    agent = PolicyGradientAgent(model, state_size=4, seed=SEED, lr=0.005)
    #load_model(model, 'last_run/solved.pth')
    train_pg(environment, agent, seed=SEED, n_episodes=4000, max_t=1000,
             solve_score=195.0,
             gamma=0.99,
             graph_when_done=False)
