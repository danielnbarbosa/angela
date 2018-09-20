#!/usr/bin/env python

import argparse
import sys
sys.path.append('env_configs/gym')
sys.path.append('env_configs/unity')
sys.path.append('env_configs/ple')

parser = argparse.ArgumentParser()
parser.add_argument("--agent", help="agent type: [dqn | hc | pg]", type=str)
parser.add_argument("--env", help="environment", type=str)
parser.add_argument("--render", help="render", type=bool, default=False)
parser.add_argument("--load", help="filename of saved model", type=str, default=None)
args = parser.parse_args()

env_config = __import__(args.env)

if args.agent == 'dqn':
    env_config.dqn(render=args.render, load_file=args.load)
elif args.agent == 'hc':
    env_config.hc(render=args.render, load_file=args.load)
elif args.agent == 'pg':
    env_config.pg(render=args.render, load_file=args.load)
