#!/bin/bash

# Wrapper script to execute a training run on all the working environments.


train_envs() {
  for environment in $1
  do
    echo "Starting training for: $environment"
    python $environment
  done
}

cd gym
#environments="acrobot.py cartpole.py frozenlake.py frozenlake8x8.py lunarlander.py mountaincar.py pendulum.py"
environments="frozenlake.py frozenlake8x8.py lunarlander.py mountaincar.py pendulum.py"
train_envs "$environments"


cd ../unity
environments="banana.py basic.py"
train_envs "$environments"
