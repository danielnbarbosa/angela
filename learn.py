#!/usr/bin/env python

from libs.environments import gym, unity, ple
import importlib
import argparse
import sys
sys.path.append('cfg')

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", help="hyperparameter config file", type=str)
parser.add_argument("--render", help="render agent", action="store_true")
parser.add_argument("--load", help="path to saved model", type=str, default=None)
args = parser.parse_args()

# load config from file
cfg = importlib.import_module(args.cfg)

# create environment
if   cfg.env_class == 'Gym':                     environment = gym.Gym(**cfg.environment)
elif cfg.env_class == 'GymAtari':                environment = gym.GymAtari(**cfg.environment)
elif cfg.env_class == 'GymAtariPong':            environment = gym.GymAtariPong(**cfg.environment)
elif cfg.env_class == 'GymAtariBreakout':        environment = gym.GymAtariBreakout(**cfg.environment)
elif cfg.env_class == 'UnityMLVector':           environment = unity.UnityMLVector(**cfg.environment)
elif cfg.env_class == 'UnityMLVectorMultiAgent': environment = unity.UnityMLVectorMultiAgent(**cfg.environment)
elif cfg.env_class == 'UnityMLVisual':           environment = unity.UnityMLVisual(**cfg.environment)
elif cfg.env_class == 'PLEFlappyBird':           environment = ple.PLEFlappyBird(render=args.render, **cfg.environment)

# based on agent_type, create model and agent then start training
if cfg.agent_type == 'dqn':
    from libs.agents.dqn import agents, models, training
    if cfg.model_class == 'TwoLayer2x':      model = models.TwoLayer2x(**cfg.model)
    elif cfg.model_class == 'Dueling2x':     model = models.Dueling2x(**cfg.model)
    elif cfg.model_class == 'Conv3D2x':      model = models.Conv3D2x(**cfg.model)
    agent = agents.DQN(model, load_file=args.load, **cfg.agent)
    training.train(environment, agent, render=args.render, **cfg.train)

elif cfg.agent_type == 'hc':
    from libs.agents.hc import agents, models, training
    if cfg.model_class == 'SingleLayerPerceptron': model = models.SingleLayerPerceptron(**cfg.model)
    agent = agents.HillClimbing(model, load_file=args.load, **cfg.agent)
    training.train(environment, agent, render=args.render, **cfg.train)

elif cfg.agent_type == 'pg':
    from libs.agents.pg import agents, models, training
    if   cfg.model_class == 'SingleHiddenLayer': model = models.SingleHiddenLayer(**cfg.model)
    elif cfg.model_class == 'TwoLayerConv2D':    model = models.TwoLayerConv2D(**cfg.model)
    elif cfg.model_class == 'Conv3D':            model = models.Conv3D(**cfg.model)
    agent = agents.PolicyGradient(model, load_file=args.load, **cfg.agent)
    training.train(environment, agent, render=args.render, **cfg.train)

elif cfg.agent_type == 'ppo':
    from libs.agents.ppo import agents, models, training
    if   cfg.model_class == 'SingleHiddenLayer': model = models.SingleHiddenLayer(**cfg.model)
    elif cfg.model_class == 'TwoLayerConv2D':    model = models.TwoLayerConv2D(**cfg.model)
    agent = agents.ProximalPolicyOptimization(model, load_file=args.load, **cfg.agent)
    training.train(environment, agent, render=args.render, **cfg.train)

elif cfg.agent_type == 'ddpg':
    from libs.agents.ddpg import agents, models, training
    if cfg.model_class == 'LowDim2x':      model = models.LowDim2x(**cfg.model)
    agent = agents.DDPG(model, load_file=args.load, **cfg.agent)
    training.train(environment, agent, render=args.render, **cfg.train)
