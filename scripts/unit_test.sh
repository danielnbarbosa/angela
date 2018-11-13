#!/bin/bash

# "Unit tests" for the various algorithms. Runs each algorithm against the CartPole environment
# to ensure that refactoring doesn't break things.  As seeds are held constant results should
# be identical.


CFG='cartpole_dqn cartpole_hc cartpole_pg cartpole_ppo'

for cfg in $CFG
do
  echo "Testing $cfg"
  ./main.py --cfg=cfg/gym/cartpole/$cfg.py | tr '\r' '\n' > scripts/unit_tests/tmp.txt
  diff scripts/unit_tests/tmp.txt scripts/unit_tests/$cfg.txt
done

rm scripts/unit_tests/tmp.txt
