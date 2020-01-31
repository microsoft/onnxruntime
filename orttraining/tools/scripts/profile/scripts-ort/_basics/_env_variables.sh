#!/bin/bash

export OMP_NUM_THREADS=1
export PHILLY_SCRIPT_ROOT="$PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/"

# for local test
# export PHILLY_SCRIPT_ROOT="/mnt/c/dev/onnxruntime/training/script/"
# export PHILLY_CONTAINER_INDEX=0

export ORT_SCRIPT_PATH="${PHILLY_SCRIPT_ROOT}profile/scripts-ort/"
export PT_SCRIPT_PATH="${PHILLY_SCRIPT_ROOT}profile/scripts-pt/"
export CUSTOM_PARAMS_STRING=""
export MPI_TYPE="philly"
export CONTAINTER_ZERO_WAIT_TIME="0s" #"2m"