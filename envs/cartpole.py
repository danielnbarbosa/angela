from imports import *

SEED = 42
#SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = GymEnvironment('CartPole-v1', seed=SEED)

model = DuelingQNet(state_size=4, action_size=2, fc1_units=64, fc2_units=32, seed=SEED)

agent = Agent(model, action_size=2, seed=SEED,
              use_double_dqn=True,
              use_prioritized_experience_replay=False,
              alpha_start=0.5,
              alpha_decay=0.9992)

#load(model, 'cartpole.pth')
train(environment, agent, n_episodes=1000, max_t=1000, solve_score=195.0)
