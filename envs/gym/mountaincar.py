from imports import *

SEED = 0
#SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = GymEnvironment('MountainCar-v0', seed=SEED, max_steps=1000)


def dqn():
    model = DuelingQNet(state_size=2, action_size=3, fc1_units=64, fc2_units=64, seed=SEED)
    agent = DQNAgent(model, action_size=3, seed=SEED,
                     use_double_dqn=True,
                     use_prioritized_experience_replay=False)
    train_dqn(environment, agent, n_episodes=4000, max_t=1000, solve_score=-110.0,
              eps_start=1.0,
              eps_end=0.05,
              eps_decay=0.997)


def hc():
    agent = HillClimbingAgent(state_size=2, action_size=3, seed=SEED, policy='deterministic')
    train_hc(environment, agent, seed=SEED, n_episodes=1000, max_t=1000,
             use_adaptive_noise=True,
             npop=4,
             print_every=20,
             solve_score=-110.0,
             graph_when_done=True)


def pg():
    model = SingleHiddenLayerWithSoftmaxOutput(state_size=2, action_size=3, fc1_units=16, seed=SEED)
    agent = PolicyGradientAgent(model, state_size=2, seed=SEED, lr=0.005)
    train_pg(environment, agent, seed=SEED, n_episodes=5000, max_t=1000,
             gamma=0.99,
             solve_score=-110.0,
             graph_when_done=True)
