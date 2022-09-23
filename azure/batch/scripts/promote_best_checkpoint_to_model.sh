#!/bin/bash

echo
echo
echo '####### SAVE AND PROMOTE BEST CHECKPOINT TO MODEL #######'
echo

set -ex

# find the lowest loss checkpoint file path (without suffix)
best_checkpoint=$(find $1 -name *.ckpt -exec basename {} .ckpt \; | sort -t . -k 2 -n | head -n 1 | xargs -I {} find $1 -name {}.ckpt)
echo "the checkpoint with the lowest loss value is: $best_checkpoint"

# create a filename that links the job to the checkpoint resutls
mkdir -p $AZ_BATCH_NODE_MOUNTS_DIR/$MODELS_CONTAINER_MOUNT_DIR/$TRAINING_JOB_RESULTS_DIR

# move the checkpoint to the models job dir for long terms storage (tracking)
cp -f $best_checkpoint $AZ_BATCH_NODE_MOUNTS_DIR/$MODELS_CONTAINER_MOUNT_DIR/$TRAINING_JOB_RESULTS_DIR/

# overwrite the existing deployed model for use in other services (prediction scripts etc)
cp -f $best_checkpoint $AZ_BATCH_NODE_MOUNTS_DIR/$MODELS_CONTAINER_MOUNT_DIR/zoobot.ckpt
