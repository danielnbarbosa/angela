from imports import *


"""
NOTE: download pre-built Unity Bannana.app from: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/
"""

def dqn():
    SEED=895815691
    #SEED = random.randint(0, 2 ** 30)
    #print('SEED: {}'.format(SEED))

    environment = UnityMLVectorEnvironment('compiled_unity_environments/Banana.app', seed=SEED)
    #environment = UnityMLEnvironment('Banana_Linux/Banana.x86_64', 'vector')

    model = TwoHiddenLayerQNet(state_size=37, action_size=4, fc1_units=32, fc2_units=32, seed=SEED)

    agent = DQNAgent(model, action_size=4, seed=SEED,
                     use_double_dqn=False,
                     use_prioritized_experience_replay=False)

    train_dqn(environment, agent, n_episodes=1000, solve_score=13.0,
              eps_start=1.0,
              eps_end=0.001,
              eps_decay=0.97)



def hc():
    SEED=895815691
    #SEED = random.randint(0, 2 ** 30)
    #print('SEED: {}'.format(SEED))

    environment = UnityMLVectorEnvironment('compiled_unity_environments/Banana.app', seed=SEED)

    agent = HillClimbingAgent(state_size=37, action_size=4, seed=SEED,
                              policy='deterministic')

    train_hc(environment, agent, seed=SEED, n_episodes=2000, solve_score=13.0,
             use_adaptive_noise=True,
             npop=6,
             print_every=20,
             graph_when_done=True)


### main ###
dqn()
#hc()
