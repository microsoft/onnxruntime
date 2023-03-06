#!/bin/bash

#set -x

WS=$(dirname $(realpath $0))

export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

export HSA_ENABLE_SDMA=0
export MIOPEN_FIND_MODE=NORMAL

BATCH=1
SEQ_LEN=128
MODEL_NAME='bert-base-cased'

#PROF="rocprof -d rocp --timestamp on --hip-trace --roctx-trace "

CMD="python bert.py --fp16 --model=$MODEL_NAME --batch=$BATCH --seq-len=$SEQ_LEN --output model-${MODEL_NAME}.onnx"


$PROF $CMD --export --loop-cnt=10
#$PROF $CMD --no-torch-infer --loop-cnt=10
