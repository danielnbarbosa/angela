from imports import *

#SEED = 194917489   # -99 episodes to solve
#SEED = 599915191   # +15 episodes to solve
SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = GymEnvironment('CartPole-v0', seed=SEED,
                             max_steps=1000)

#model = DuelingQNet(state_size=4, action_size=2, fc1_units=64, fc2_units=32, seed=SEED)

agent = HillClimbingAgent(state_size=4, action_size=2, seed=SEED,
                          policy='deterministic')

train_hc(environment, agent, seed=SEED, n_episodes=1000, max_t=1000,
         print_every=1,
         render_every=1000,
         solve_score=195.0)
