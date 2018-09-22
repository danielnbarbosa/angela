#!/usr/bin/env python

from libs import environments, models, agents, train
from libs.agents import dqn, hc, pg
from libs.models import dqn, hc, pg
from libs.environments import gym, ple, unity
import argparse
import sys
sys.path.append('cfg')

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", help="hyperparameter config file", type=str)
parser.add_argument("--render", help="render agent", action="store_true")
parser.add_argument("--load", help="path to saved model", type=str, default=None)
args = parser.parse_args()

# load config from file
cfg = __import__(args.cfg)

# create environment
if   cfg.env_class == 'Gym':           environment = environments.gym.Gym(**cfg.environment)
elif cfg.env_class == 'GymAtari':      environment = environments.gym.GymAtari(**cfg.environment)
elif cfg.env_class == 'GymAtariPong':  environment = environments.gym.GymAtariPong(**cfg.environment)
elif cfg.env_class == 'UnityMLVector': environment = environments.unity.UnityMLVector(**cfg.environment)
elif cfg.env_class == 'UnityMLVisual': environment = environments.unity.UnityMLVisual(**cfg.environment)
elif cfg.env_class == 'PLEFlappyBird': environment = environments.ple.PLEFlappyBird(render=args.render, **cfg.environment)

# based on agent_type, create model and agent then start training
if cfg.agent_type == 'dqn':
    if cfg.model_class == 'TwoLayer2x':      model = models.dqn.TwoLayer2x(**cfg.model)
    elif cfg.model_class == 'Dueling2x':     model = models.dqn.Dueling2x(**cfg.model)
    elif cfg.model_class == 'Conv3D2x':      model = models.dqn.Conv3D2x(**cfg.model)
    agent = agents.dqn.DQN(model, load_file=args.load, **cfg.agent)
    train.dqn(environment, agent, render=args.render, **cfg.train)

elif cfg.agent_type == 'hc':
    if cfg.model_class == 'SingleLayerPerceptron': model = models.hc.SingleLayerPerceptron(**cfg.model)
    agent = agents.hc.HillClimbing(model, load_file=args.load, **cfg.agent)
    train.hc(environment, agent, render=args.render, **cfg.train)

elif cfg.agent_type == 'pg':
    if   cfg.model_class == 'SingleHiddenLayer': model = models.pg.SingleHiddenLayer(**cfg.model)
    elif cfg.model_class == 'TwoLayerConv2D':    model = models.pg.TwoLayerConv2D(**cfg.model)
    elif cfg.model_class == 'Conv3D':            model = models.pg.Conv3D(**cfg.model)
    agent = agents.pg.PolicyGradient(model, load_file=args.load, **cfg.agent)
    train.pg(environment, agent, render=args.render, **cfg.train)
