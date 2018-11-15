#!/usr/bin/env python

import importlib
import argparse
import sys
import signal
from subprocess import call

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", help="path to hyperparameter config file", type=str)
parser.add_argument("--render", help="render agent", action="store_true")
parser.add_argument("--load", help="path to saved model", type=str, default=None)
args = parser.parse_args()

# load config from file
sys.path.append('/'.join(args.cfg.split('/')[0:3]))
cfg = importlib.import_module(args.cfg.split('/')[-1].split('.')[0])


# load proper environment module and create environment object
if cfg.env_class.startswith('Gym'):
    from libs.environments import gym
    environment = eval('gym.' + cfg.env_class + '(**cfg.environment)')
elif cfg.env_class.startswith('Unity'):
    from libs.environments import unity
    environment = eval('unity.' + cfg.env_class + '(**cfg.environment)')
elif cfg.env_class.startswith('PLE'):
    from libs.environments import ple
    environment = eval('ple.' + cfg.env_class + '(render=args.render, **cfg.environment)')

# load modules based on algorithm
exec('from libs.algorithms.' + cfg.algorithm + ' import agents, models, training')

# trap Ctrl-C
def signal_handler(sig, frame):
        call('killall fceux 2> /dev/null', shell=True)  # shutdown NES emulator
        call('killall fceux 2> /dev/null', shell=True)
        environment.env.close()
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

# create model and agent objects and start training
if cfg.algorithm == 'maddpg_v2':  # model is loaded from inside the agent
    agent = agents.Agent(load_file=args.load, **cfg.agent)
    training.train(environment, agent, render=args.render, **cfg.train)
else:
    model = eval('models.' + cfg.model_class + '(**cfg.model)')
    agent = agents.Agent(model, load_file=args.load, **cfg.agent)
    training.train(environment, agent, render=args.render, **cfg.train)
