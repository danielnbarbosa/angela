#!/bin/bash

# Backup weights from training run into named directory

DIR=$1

cp -r checkpoints/last_run checkpoints/$DIR
rm checkpoints/$DIR/README.md
