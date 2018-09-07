#!/bin/bash
EXE="eager_hadmult_simple.py"

BATCH_SIZE=100
MODELDIR="/tmp/simple_hadmult_eager"
DATA_DIR="/Users/perdue/Dropbox/Data/RandomData/hdf5"
TRAIN_FILE="${DATA_DIR}/hadmultkineimgs_mnvvtx_train.hdf5"
EVAL_FILE="${DATA_DIR}/hadmultkineimgs_mnvvtx_test.hdf5"

ARGS="--batch-size ${BATCH_SIZE}"
ARGS+=" --model-dir ${MODELDIR}"
ARGS+=" --train-file ${TRAIN_FILE}"
ARGS+=" --eval-file ${EVAL_FILE}"

cat << EOF
python $EXE $ARGS
EOF
python $EXE $ARGS
