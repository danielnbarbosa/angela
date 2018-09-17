#!/bin/bash

# Wrapper script to execute a training run on all the working environments.

ENVIRONMENTS='cartpole frozenlake frozenlake8x8 acrobot mountaincar pendulum lunarlander banana basic'
AGENTS='dqn hc pg'


for environment in $ENVIRONMENTS
do
  for agent in $AGENTS
  do
    echo ''
    echo '----------------------------------------------------------------'
    echo "Starting training on $environment environment with $agent agent."
    ./train.py --env $environment --agent $agent
  done
done
