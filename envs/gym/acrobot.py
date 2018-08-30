from imports import *

SEED = 0
#SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = GymEnvironment('Acrobot-v1', seed=SEED)

model = DuelingQNet(state_size=6, action_size=3, fc1_units=64, fc2_units=64, seed=SEED)

agent = DQNAgent(model, action_size=3, seed=SEED,
              use_double_dqn=True,
              use_prioritized_experience_replay=False)

train_dqn(environment, agent, n_episodes=12000, max_t=1000)
