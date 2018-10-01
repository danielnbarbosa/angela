#!/bin/bash

IP=107.23.183.24
DIR=$1

mkdir -p checkpoints/$DIR
scp ubuntu@$IP:angela/checkpoints/last_run/* checkpoints/$DIR/
