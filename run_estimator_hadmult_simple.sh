#!/bin/bash
EXE="estimator_hadmult_simple.py"
BATCH_SIZE=100
TRAIN_STEPS=5
NUM_EPOCHS=1

MODEL_DIR="/tmp/simple_hadmult_estimator"
DATA_DIR="/Users/perdue/Dropbox/Data/RandomData/hdf5"
TRAIN_FILE="${DATA_DIR}/hadmultkineimgs_mnvvtx_train.hdf5"
EVAL_FILE="${DATA_DIR}/hadmultkineimgs_mnvvtx_test.hdf5"

ARGS="--batch-size ${BATCH_SIZE}"
ARGS+=" --train-steps ${TRAIN_STEPS}"
ARGS+=" --num-epochs ${NUM_EPOCHS}"
ARGS+=" --train-file ${TRAIN_FILE}"
ARGS+=" --eval-file ${EVAL_FILE}"
ARGS+=" --model-dir ${MODEL_DIR}"

cat << EOF
python $EXE $ARGS
EOF
python $EXE $ARGS
