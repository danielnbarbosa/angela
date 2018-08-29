from imports import *

SEED = 42
#SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = GymEnvironment('FrozenLake-v0', seed=SEED, one_hot=16)

model = DuelingQNet(state_size=16, action_size=4, fc1_units=32, fc2_units=32, seed=SEED)

agent = Agent(model, action_size=4, seed=SEED,
              use_double_dqn=True,
              use_prioritized_experience_replay=False)

train(environment, agent, n_episodes=4000, max_t=1000)
