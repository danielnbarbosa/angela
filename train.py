#!/usr/bin/env python

import argparse
import sys
sys.path.insert(0, 'libs')
sys.path.insert(1, 'envs/gym')
sys.path.insert(2, 'envs/unity')


parser = argparse.ArgumentParser()
parser.add_argument("--agent", help="agent type: [dqn | hc | pg]", type=str)
parser.add_argument("--env", help="environment", type=str)
parser.add_argument("--render", help="render", type=bool, default=False)
args = parser.parse_args()

env = __import__(args.env)

if args.agent == 'dqn':
    env.dqn(render=args.render)
elif args.agent == 'hc':
    env.hc(render=args.render)
elif args.agent == 'pg':
    env.pg(render=args.render)
