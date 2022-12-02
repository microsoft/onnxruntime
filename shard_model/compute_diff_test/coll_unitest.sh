#!/bin/bash

WS=$(dirname $(realpath $0))

NUM_GPUS=4
TYPE='float16'
#TYPE='float32'

MPI="mpirun --tag-output --npernode $NUM_GPUS --bind-to numa "

#MODE='allreduce'
MODE='allgather'

GEN_CMD="python ${MODE}_unitest.py --generate-model --type=$TYPE --size=$NUM_GPUS"

CMD="python ${MODE}_unitest.py --type=$TYPE --size=$NUM_GPUS"

TORCHCMD="python torch_allreduce_test.py --type=$TYPE --size=$NUM_GPUS"

$GEN_CMD

$MPI $CMD

#$MPI $TORCHCMD
