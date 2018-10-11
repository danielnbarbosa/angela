#!/bin/bash

# Backup weights from training run into named directory

DIR=$1

cd ..
cp -r checkpoints/last_run checkpoints/$DIR
rm checkpoints/$DIR/README.md
