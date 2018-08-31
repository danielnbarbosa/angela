from imports import *


def dqn():
    SEED = 42
    #SEED = random.randint(0, 2 ** 30)
    #print('SEED: {}'.format(SEED))

    environment = GymEnvironment('Pendulum-v0', seed=SEED, max_steps=1000, action_bins=(10,))

    model = DuelingQNet(state_size=3, action_size=9, fc1_units=32, fc2_units=32, seed=SEED)

    agent = DQNAgent(model, action_size=9, seed=SEED,
                  use_double_dqn=True,
                  use_prioritized_experience_replay=False)

    train_dqn(environment, agent, n_episodes=4000, max_t=1000)


def hc():
    # SEED = 875812345
    SEED = random.randint(0, 2 ** 30)
    print('SEED: {}'.format(SEED))

    environment = GymEnvironment('Pendulum-v0', seed=SEED, max_steps=1000, action_bins=(10,))

    agent = HillClimbingAgent(state_size=3, action_size=9, seed=SEED,
                              policy='deterministic')

    train_hc(environment, agent, seed=SEED, n_episodes=1000, max_t=1000,
             use_adaptive_noise=True,
             npop=10,
             print_every=100,
             graph_when_done=True)


### main ###
#dqn()
hc()
