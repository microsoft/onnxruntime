#!/bin/bash

NUM_GPUS=${1:-1}

MPI="mpirun --allow-run-as-root
    -mca btl_openib_warn_no_device_params_found 0 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0
    --tag-output --npernode $NUM_GPUS --bind-to numa
    -x MIOPEN_FIND_MODE=1"

CMD="$MPI python convert_to_onnx.py ${@:2}"

$CMD