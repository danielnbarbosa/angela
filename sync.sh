#!/bin/bash

# Sync weights from last run in AWS to local.
# Example: ./sync.sh 54.161.76.124 pong_ppo_gpu_03

IP=$1
DIR=$2

mkdir -p checkpoints/$DIR
scp ubuntu@$IP:angela/checkpoints/last_run/* checkpoints/$DIR/
rm checkpoints/$DIR/README.md
