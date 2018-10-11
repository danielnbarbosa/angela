#!/bin/bash

# Wrapper script to execute a training run on a variety of simple environments.

CFGS='cartpole_dqn cartpole_hc cartpole_pg
      frozenlake_dqn frozenlake8x8_dqn
      acrobot_dqn acrobot_hc acrobot_pg
      mountaincar_dqn mountaincar_hc mountaincar_pg
      pendulum_dqn pendulum_hc pendulum_pg
      lunarlander_dqn lunarlander_hc lunarlander_pg
      basic_dqn'

cd ..
for cfg in $CFGS
do
  echo ''
  echo '----------------------------------------------------------------'
  echo "| Starting training on $cfg configuration."
  echo '----------------------------------------------------------------'
  ./learn.py --cfg $cfg
done
