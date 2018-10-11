#!/bin/bash

# Sync weights from AWS and run inference on them.
# Example: ./inference.sh 54.161.76.124 episode.3800
# Useful for testing locally a Crawler agent trained on AWS

IP=$1
FILE_PREFIX=$2
LOCAL_DIR=crawler_ddpg_aws
CFG=crawler_ddpg

scp ubuntu@$IP:angela/checkpoints/last_run/$FILE_PREFIX.*  checkpoints/$LOCAL_DIR/
./learn.py --cfg $CFG --load=checkpoints/$LOCAL_DIR/$FILE_PREFIX
