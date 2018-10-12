#!/bin/bash

# Sync weights from AWS and run evaluation on them.
# Example: ./evaluate.sh 54.161.76.124 bipedalwalker_ddpg bipedalwalker_ddpg_evaluation episode.1000
# Useful for training an agent on AWS and evaluating it locally

IP=$1
LOCAL_DIR=$2
CFG=$3
FILE_PREFIX=$4

cd ..
mkdir -p checkpoints/$LOCAL_DIR/
scp ubuntu@$IP:angela/checkpoints/last_run/$FILE_PREFIX.*  checkpoints/$LOCAL_DIR/
./learn.py --cfg $CFG --render --load=checkpoints/$LOCAL_DIR/$FILE_PREFIX
