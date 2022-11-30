#!/bin/bash
set -x

NUM_GPUS=4
#DTYPE='fp32'
DTYPE='fp16'
SEQ_LEN=512
BATCH=1024

ORI_MODEL="onnx_models/bert_base_cased_1_${DTYPE}_gpu.onnx"
SHAPED_MODEL="bert_base_cased_1_${DTYPE}_gpu_shaped.onnx"
PREFIX="bert_shard_megatron-${DTYPE}"
#PREFIX="bert_shard_allgather-${DTYPE}"

MPI="mpirun -mca btl_openib_warn_no_device_params_found 0 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 --tag-output --npernode $NUM_GPUS --bind-to numa "

GETMODEL="python -m onnxruntime.transformers.benchmark -g -m bert-base-cased --sequence_length ${SEQ_LEN} --batch_sizes ${BATCH} --provider=cuda -p fp16 --disable_embed_layer_norm"

PROCESS_SHAPE="python process_shape.py --input=$ORI_MODEL --output=$SHAPED_MODEL --batch=$BATCH --seq-len=$SEQ_LEN"

SHARD_CMD="python shard.py --model-file=${SHAPED_MODEL} --shard-prefix=${PREFIX}"

COMPARE_CMD="python compare.py --model-file=${ORI_MODEL} --shard-prefix=${PREFIX}"

BENCHMARK="python benchmark.py --model-file=${ORI_MODEL} --seq-len=$SEQ_LEN --batch=$BATCH --loop-cnt=1000"

SHARD_BENCHMARK="python benchmark.py --shard-prefix=${PREFIX} --seq-len=$SEQ_LEN --batch=$BATCH --loop-cnt=1000"


# working piplines

$GETMODEL

#$PROCESS_SHAPE
#$SHARD_CMD
#$MPI $COMPARE_CMD
$BENCHMARK
#$MPI $SHARD_BENCHMARK
