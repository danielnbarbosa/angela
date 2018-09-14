from imports import *


def dqn():
    SEED = 0
    #SEED = random.randint(0, 2 ** 30)
    #print('SEED: {}'.format(SEED))

    environment = GymEnvironment('CartPole-v1', seed=SEED)

    model = DuelingQNet(state_size=4, action_size=2, fc1_units=64, fc2_units=32, seed=SEED)

    agent = DQNAgent(model, action_size=2, seed=SEED,
                  use_double_dqn=True,
                  use_prioritized_experience_replay=False,
                  alpha_start=0.5,
                  alpha_decay=0.9992,
                  buffer_size=10000)

    train_dqn(environment, agent, n_episodes=1000, max_t=1000, solve_score=195.0)


def hc():
    #SEED = 878833714   # -99 episodes to solve (with adaptive noise)
    #SEED = 256533649   # +96 episodes to solve (with adaptive noise)
    #SEED = 983301353   # good for seeing the difference between having adaptive noise and not
    SEED = 897277145
    #SEED = random.randint(0, 2 ** 30)
    #print('SEED: {}'.format(SEED))

    environment = GymEnvironment('CartPole-v1', seed=SEED,
                                 max_steps=1000)

    agent = HillClimbingAgent(state_size=4, action_size=2, seed=SEED,
                              policy='deterministic')

    #load_pickle(agent, 'last_run/solved.pck')
    train_hc(environment, agent, seed=SEED, n_episodes=4000, max_t=1000,
             use_adaptive_noise=False,
             npop=10,
             print_every=10,
             solve_score=195.0,
             graph_when_done=False)


def pg():
    SEED = 0
    #SEED = random.randint(0, 2 ** 30)
    #print('SEED: {}'.format(SEED))

    environment = GymEnvironment('CartPole-v1', seed=SEED,
                                 max_steps=1000)

    model = SingleHiddenLayerWithSoftmaxOutput(state_size=4, action_size=2, fc1_units=16, seed=SEED)

    agent = PolicyGradientAgent(model, state_size=4, seed=SEED,
                                lr=0.005)

    train_pg(environment, agent, seed=SEED, n_episodes=4000, max_t=1000,
             solve_score=195.0,
             gamma=0.99,
             graph_when_done=False)

### main ###
#dqn()
#hc()
pg()
