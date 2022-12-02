#!/bin/bash

#export ORT_CUDA_GEMM_OPTIONS=1  # fp16 compute type
WS=$(dirname $(realpath $0))

NUM_GPUS=4
M=512
K=768
N=1024
TYPE='float16'
#TYPE='float32'

MODE='split_n'
#MODE='split_k'

CMD="python test_${MODE}_matmul.py --type=$TYPE --m=$M --n=$N --k=$K --size=$NUM_GPUS"

$CMD

