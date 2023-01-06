#!/bin/bash

NUM_GPUS=2

TORCHRUN="torchrun --nproc_per_node=$NUM_GPUS "

MPI="mpirun -mca btl_openib_warn_no_device_params_found 0 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 --tag-output --npernode $NUM_GPUS --bind-to numa "

MODEL_NAME='bloom'

CMD="bloom.py --output saved-model-${MODEL_NAME}.onnx"

#python $CMD
#$TORCHRUN $CMD
$MPI python $CMD
