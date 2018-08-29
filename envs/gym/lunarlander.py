from imports import *


SEED = 42
#SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = GymEnvironment('LunarLander-v2', seed=SEED)

model = DuelingQNet(state_size=8, action_size=4, fc1_units=64, fc2_units=64, seed=SEED)

agent = Agent(model, action_size=4, seed=SEED,
              use_double_dqn=True,
              use_prioritized_experience_replay=False,
              buffer_size=100000)

#load(model, 'lunarlander.pth')
train(environment, agent, n_episodes=4000, max_t=2000, solve_score=200.0)
