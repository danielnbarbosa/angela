from imports import *

SEED = 42
#SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = GymEnvironment('Pendulum-v0', seed=SEED, max_steps=1000, action_bins=(10,))

model = DuelingQNet(state_size=3, action_size=9, fc1_units=32, fc2_units=32, seed=SEED)

agent = Agent(model, action_size=9, seed=SEED,
              use_double_dqn=True,
              use_prioritized_experience_replay=False)

train(environment, agent, n_episodes=4000, max_t=1000)
