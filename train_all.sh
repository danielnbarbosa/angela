#!/bin/bash

# Wrapper script to execute a training run on all the working environments.


train_envs() {
  for environment in $1
  do
    echo "Starting training for: $environment"
    ./train.py --env $environment --agent dqn
  done
}

#environments="acrobot.py cartpole.py frozenlake.py frozenlake8x8.py lunarlander.py mountaincar.py pendulum.py"
environments="acrobot cartpole frozenlake frozenlake8x8 lunarlander mountaincar pendulum banana basic"
train_envs "$environments"
