#!/bin/bash

MODEL_NAME=${LLAMA2_MODEL_NAME:-"Llama-2-7b-hf"}

OUTPUT=$MODEL_NAME
# OUTPUT="/workspace/work/llama2/Llama-2-7b-hf"

CMD="python llama-v2.py --model=meta-llama/$MODEL_NAME --output-name=$OUTPUT ${@}"

# if [[ $OMPI_COMM_WORLD_LOCAL_RANK -eq 0 ]] ; then
#     PROF="rocprof --timestamp on --hip-trace --roctx-trace"
#     # PROF="rocprof -d rocp --timestamp on --hip-trace --roctx-trace"
#     # PROF=""
#     # export NCCL_DEBUG=INFO
#     # export NCCL_DEBUG_SUBSYS=COLL
#     CMD="$PROF $CMD"
# fi

set -x
$CMD

# --export  # --verbose
# --generate --torch  # --compile
# --generate --ort
# --benchmark --torch --loop-cnt=2 --warm=1  # --compile
# --benchmark --tunable --tuning --ort --loop-cnt=5 --warm=1
