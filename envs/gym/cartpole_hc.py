from imports import *

#SEED = 878833714   # -99 episodes to solve (with adaptive noise)
#SEED = 256533649   # +96 episodes to solve (with adaptive noise)
#SEED = 983301353   # good for seeing the difference between having adaptive noise and not
SEED = random.randint(0, 2 ** 30)
#print('SEED: {}'.format(SEED))

environment = GymEnvironment('CartPole-v0', seed=SEED,
                             max_steps=1000)

agent = HillClimbingAgent(state_size=4, action_size=2, seed=SEED,
                          policy='deterministic')

train_hc(environment, agent, seed=SEED, n_episodes=4000, max_t=1000,
         use_adaptive_noise=True,
         print_every=1,
         render_every=10000,
         solve_score=195.0,
         graph_when_done=True)
