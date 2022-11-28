#!/bin/bash
set -x

NUM_GPUS=4
#DTYPE='fp32'
DTYPE='fp16'
#MODEL="bert_base_cased_1_${DTYPE}_gpu_shaped.onnx"
MODEL='onnx_models/bert_base_cased_1_fp16_gpu.onnx'
PREFIX="bert_shard_megatron-${DTYPE}"
#PREFIX="bert_shard_allgather-${DTYPE}"
SEQ_LEN=128
BATCH=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL

#PROF="nsys profile -o dist-bert-infer -f true --trace=cuda,nvtx,cublas,cudnn "

MPI="mpirun -mca btl_openib_warn_no_device_params_found 0 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 --tag-output --npernode $NUM_GPUS --bind-to numa "

CMD="python run_model.py --model-file=${MODEL} --shard-prefix=${PREFIX} --benchmark"

BENCHMARK="python benchmark.py --model-file=${MODEL} --seq-len=$SEQ_LEN --batch=$BATCH --loop-cnt=10"
SHARD_BENCHMARK="python benchmark.py --shard-prefix=${PREFIX} --seq-len=$SEQ_LEN --batch=$BATCH --loop-cnt=2"

#python shard.py --model-file=${MODEL} --shard-prefix=${PREFIX}

#$MPI $CMD

#$PROF $BENCHMARK

$PROF $MPI $SHARD_BENCHMARK
