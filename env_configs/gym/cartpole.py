from libs import environments, models, agents, train

SEED = 0
#SEED = random.randint(0, 2 ** 30)

environment = environments.Gym('CartPole-v1', seed=SEED, max_steps=1000)
state_size = 4
action_size = 2

def dqn(render, load_file):
    model_config = {
        'fc_units': (64, 32)
    }
    agent_config = {
        'use_double_dqn': True,
        'use_prioritized_experience_replay': False,
        'update_every': 4,
        'lr': 0.0006,
        'alpha_start': 0.5,
        'alpha_decay': 0.9992,
        'buffer_size': 100000
    }
    train_config = {
        'n_episodes': 1000,
        'max_t': 1000,
        'solve_score': 195.0,
        'eps_start': 1.0,
        'eps_end': 0.01,
        'eps_decay': 0.995
    }

    model = models.dqn.Dueling2x(state_size, action_size, seed=SEED, **model_config)
    agent = agents.DQN(model, action_size, seed=SEED, load_file=load_file, **agent_config)
    train.dqn(environment, agent, render=render, **train_config)


def hc(render, load_file):
    model_config = {
    }
    agent_config = {
        'use_adaptive_noise': False,
        'policy': 'deterministic'
    }
    train_config = {
        'n_episodes': 4000,
        'max_t': 1000,
        'solve_score': 195.0,
        'npop': 10
    }

    model = models.hc.SingleLayerPerceptron(state_size, action_size, seed=SEED, **model_config)
    agent = agents.HillClimbing(model, action_size, seed=SEED, load_file=load_file, **agent_config)
    train.hc(environment, agent, seed=SEED, render=render, **train_config)


def pg(render, load_file):
    model_config = {
        'fc1_units': 16
    }
    agent_config = {
        'lr': 0.005,
    }
    train_config = {
        'n_episodes': 4000,
        'max_t': 1000,
        'solve_score': 195.0,
        'gamma': 0.99
    }

    model = models.pg.SingleHiddenLayer(state_size, action_size, seed=SEED, **model_config)
    agent = agents.PolicyGradient(model, seed=SEED, load_file=load_file, **agent_config)
    train.pg(environment, agent, render=render, **train_config)
