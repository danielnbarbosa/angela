#!/bin/bash

# Log important data from latest run into log dir.
# Includes: hyperparameter config, saved weights, console logging, tensorboard logging
# Example: ./scripts/log.sh lunarlander_ppo

CFG_FILE=$1
CFG_DIR=$(echo $CFG_FILE | awk -F '_' '{print $1}')
TS_DIR=$(date '+%Y_%m_%d_%H_%M')
WEIGHTS=$(ls -t checkpoints/last_run/)
LOG_DIR="logs/$CFG_FILE/$TS_DIR"

mkdir -p $LOG_DIR
# save current hyperparameter config
cp cfg/$CFG_DIR/${CFG_FILE}.py $LOG_DIR/config.py
# save last two model weights
for WEIGHT in $(echo $WEIGHTS | awk '{print $1, $2}')
do
  cp checkpoints/last_run/$WEIGHT $LOG_DIR/
done
# save last tensorboard logging
cp -r runs/$(ls -1t runs | head -1) $LOG_DIR/
# save console output logging
cp output.txt $LOG_DIR/
