#!/bin/bash
EXE="estimator_hadmult_simple.py"
BATCH_SIZE=100
TRAIN_STEPS=5
NUM_EPOCHS=1
MODELDIR="/tmp/simple_hadmult_estimator"

DATA_DIR="/Users/perdue/Dropbox/Data/RandomData/hdf5"
TFRECORD=""

ARGS="--batch-size ${BATCH_SIZE} --train-steps ${TRAIN_STEPS} --num-epochs
${NUM_EPOCHS} --data-dir ${DATA_DIR} $TFRECORD --model-dir $MODELDIR"

python $EXE $ARGS
