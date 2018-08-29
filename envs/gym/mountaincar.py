from imports import *

SEED = 42
#SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = GymEnvironment('MountainCar-v0', seed=SEED, max_steps=1000)

model = DuelingQNet(state_size=2, action_size=3, fc1_units=64, fc2_units=64, seed=SEED)

agent = Agent(model, action_size=3, seed=SEED,
              use_double_dqn=True,
              use_prioritized_experience_replay=False)

#load(model, 'mountaincar-run2.pth')
train(environment, agent, n_episodes=4000, max_t=1000, solve_score=-110.0)
