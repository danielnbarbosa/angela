from imports import *

def dqn():
    SEED = 0
    #SEED = random.randint(0, 2 ** 30)
    #print('SEED: {}'.format(SEED))

    environment = GymEnvironment('Acrobot-v1', seed=SEED)

    model = DuelingQNet(state_size=6, action_size=3, fc1_units=64, fc2_units=64, seed=SEED)

    agent = DQNAgent(model, action_size=3, seed=SEED,
                  use_double_dqn=True,
                  use_prioritized_experience_replay=False)

    train_dqn(environment, agent, n_episodes=12000, max_t=1000)


def hc():
    #SEED = 280797091
    SEED = random.randint(0, 2 ** 30)
    print('SEED: {}'.format(SEED))

    environment = GymEnvironment('Acrobot-v1', seed=SEED)

    agent = HillClimbingAgent(state_size=6, action_size=3, seed=SEED,
                              policy='deterministic')

    train_hc(environment, agent, seed=SEED, n_episodes=5000, max_t=1000,
             use_adaptive_noise=True,
             print_every=100,
             render_every=10000,
             graph_when_done=True)


### main ###
#dqn()
hc()
