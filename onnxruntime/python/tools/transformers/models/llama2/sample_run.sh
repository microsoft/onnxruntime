#!/bin/bash

#NUM_GPUS=${1:-1}
NUM_GPUS=${1:-16}

MPI="mpirun --allow-run-as-root
    -mca btl_openib_warn_no_device_params_found 0 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0
    --tag-output --npernode $NUM_GPUS --bind-to numa
    -x MIOPEN_FIND_MODE=1"

# MPI+=" -x TRANSFORMERS_OFFLINE=1"
MPI+=" -x TRANSFORMERS_CACHE=/hf_cache"
MPI+=" -x NCCL_DEBUG=VERSION"

MODEL_NAME=${LLAMA2_MODEL_NAME:-"Llama-2-70b-hf"}
OUTPUT=$MODEL_NAME

CMD="$MPI python llama-v2.py --model=meta-llama/$MODEL_NAME --output-name=$OUTPUT ${@:2}"

set -x
$CMD --torch --generate
