#!/bin/bash

# Sync weights from AWS and run inference on them.
# Example: ./inference.sh 54.161.76.124 bipedalwalker_ddpg bipedalwalker_ddpg_inference episode.1000
# Useful for training an agent on AWS and visualzing it locally

IP=$1
LOCAL_DIR=$2
CFG=$3
FILE_PREFIX=$4

cd ..
mkdir -p checkpoints/$LOCAL_DIR/
scp ubuntu@$IP:angela/checkpoints/last_run/$FILE_PREFIX.*  checkpoints/$LOCAL_DIR/
./learn.py --cfg $CFG --render --load=checkpoints/$LOCAL_DIR/$FILE_PREFIX
