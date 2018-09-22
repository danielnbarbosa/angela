#!/usr/bin/env python

from libs import environments, agents, train
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
if   cfg.env_class == 'Gym':           environment = environments.Gym(**cfg.environment)
elif cfg.env_class == 'GymAtari':      environment = environments.GymAtari(**cfg.environment)
elif cfg.env_class == 'GymAtariPong':  environment = environments.GymAtariPong(**cfg.environment)
elif cfg.env_class == 'UnityMLVector': environment = environments.UnityMLVector(**cfg.environment)
elif cfg.env_class == 'UnityMLVisual': environment = environments.UnityMLVisual(**cfg.environment)
elif cfg.env_class == 'PLEFlappyBird': environment = environments.PLEFlappyBird(render=args.render, **cfg.environment)

# based on agent_type, create model and agent then start training
if cfg.agent_type == 'dqn':
    import libs.models.dqn
    if cfg.model_class == 'TwoLayer2x':      model = libs.models.dqn.TwoLayer2x(**cfg.model)
    elif cfg.model_class == 'Dueling2x':     model = libs.models.dqn.Dueling2x(**cfg.model)
    elif cfg.model_class == 'Conv3D2x':      model = libs.models.dqn.Conv3D2x(**cfg.model)
    agent = agents.DQN(model, load_file=args.load, **cfg.agent)
    libs.train.dqn(environment, agent, render=args.render, **cfg.train)

elif cfg.agent_type == 'hc':
    import libs.models.hc
    if cfg.model_class == 'SingleLayerPerceptron': model = libs.models.hc.SingleLayerPerceptron(**cfg.model)
    agent = agents.HillClimbing(model, load_file=args.load, **cfg.agent)
    libs.train.hc(environment, agent, render=args.render, **cfg.train)

elif cfg.agent_type == 'pg':
    import libs.models.pg
    if   cfg.model_class == 'SingleHiddenLayer': model = libs.models.pg.SingleHiddenLayer(**cfg.model)
    elif cfg.model_class == 'TwoLayerConv2D':    model = libs.models.pg.TwoLayerConv2D(**cfg.model)
    elif cfg.model_class == 'Conv3D':            model = libs.models.pg.Conv3D(**cfg.model)
    agent = libs.agents.PolicyGradient(model, load_file=args.load, **cfg.agent)
    libs.train.pg(environment, agent, render=args.render, **cfg.train)
