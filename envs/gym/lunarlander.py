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
    #SEED = 280797091
    SEED = random.randint(0, 2 ** 30)
    print('SEED: {}'.format(SEED))

    environment = GymEnvironment('LunarLander-v2', seed=SEED)

    agent = HillClimbingAgent(state_size=8, action_size=4, seed=SEED,
                              policy='deterministic')

    train_hc(environment, agent, seed=SEED, n_episodes=5000, max_t=2000,
             use_adaptive_noise=True,
             print_every=100,
             render_every=10000,
             graph_when_done=True)


### main ###
#dqn()
hc()
