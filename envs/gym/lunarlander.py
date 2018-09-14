from imports import *


def dqn():
    SEED = 42
    #SEED = random.randint(0, 2 ** 30)
    #print('SEED: {}'.format(SEED))

    environment = GymEnvironment('LunarLander-v2', seed=SEED)

    model = DuelingQNet(state_size=8, action_size=4, fc1_units=128, fc2_units=128, seed=SEED)

    agent = DQNAgent(model, action_size=4, seed=SEED,
              use_double_dqn=True,
              use_prioritized_experience_replay=False,
              buffer_size=100000)

    train_dqn(environment, agent, n_episodes=4000, max_t=2000, solve_score=200.0)


def hc():
    SEED = 888417152
    #SEED = random.randint(0, 2 ** 30)
    #print('SEED: {}'.format(SEED))

    environment = GymEnvironment('LunarLander-v2', seed=SEED)

    agent = HillClimbingAgent(state_size=8, action_size=4, seed=SEED,
                              policy='deterministic')

    train_hc(environment, agent, seed=SEED, n_episodes=1500, max_t=2000,
             use_adaptive_noise=True,
             npop=10,
             print_every=100,
             solve_score=200.0,
             graph_when_done=True)

def pg():
    SEED = 0
    #SEED = random.randint(0, 2 ** 30)
    #print('SEED: {}'.format(SEED))

    environment = GymEnvironment('LunarLander-v2', seed=SEED)

    model = SingleHiddenLayerWithSoftmaxOutput(state_size=8, action_size=4, fc1_units=32, seed=SEED)

    agent = PolicyGradientAgent(model, state_size=8, seed=SEED,
                                lr=0.005)

    train_pg(environment, agent, seed=SEED, n_episodes=5000, max_t=2000,
             gamma=0.99,
             graph_when_done=True)

### main ###
#dqn()
#hc()
pg()
