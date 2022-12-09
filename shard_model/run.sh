#!/bin/bash
#set -x

NUM_GPUS=4
#DTYPE='fp32'
DTYPE='fp16'
SEQ_LEN=768
PAST_SEQ_LEN=256
BATCH=8
MODE='allreduce'
MODEL_NAME='gpt2-large'

#ORI_MODEL="onnx_models/bert_base_cased_1_${DTYPE}_gpu.onnx"
#SHAPED_MODEL="bert_base_cased_1_${DTYPE}_gpu_shaped.onnx"
ORI_MODEL="onnx_models/gpt2-large_past_fp16/gpt2-large_past_fp16.onnx"
ORI_SHAPED_MODEL="onnx_models/gpt2-large_past_fp16/gpt2-large_past_fp16_infer_shaped.onnx"
SHAPED_MODEL="gpt2-large_past_fp16_b${BATCH}_sl${SEQ_LEN}_p${PAST_SEQ_LEN}_shaped.onnx"

PREFIX="${MODEL_NAME}_shard_$MODE-${DTYPE}"

MPI="mpirun -mca btl_openib_warn_no_device_params_found 0 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 --tag-output --npernode $NUM_GPUS --bind-to numa "

#GETMODEL="python -m onnxruntime.transformers.benchmark -g -m bert-base-cased --sequence_length ${SEQ_LEN} --batch_sizes ${BATCH} --provider=cuda -p fp16 --disable_embed_layer_norm"

GETGPT2="python -m onnxruntime.transformers.models.gpt2.benchmark_gpt2 -m ${MODEL_NAME} -v -o --use_gpu -p fp16 --sequence_lengths ${SEQ_LEN} --batch_sizes ${BATCH}"

INFER_SHAPE="python symbolic_shape_infer.py --input=$ORI_MODEL --output=$ORI_SHAPED_MODEL --save_as_external_data --all_tensors_to_one_file"

PROCESS_SHAPE="python process_shape.py --input=$ORI_SHAPED_MODEL --output=$SHAPED_MODEL --batch=$BATCH --seq-len=$SEQ_LEN --past-seq-len=${PAST_SEQ_LEN}"

PREPARE_INPUTS="python prepare_inputs.py --batch=$BATCH --seq-len=$SEQ_LEN --past-seq-len=${PAST_SEQ_LEN} --model-name=$MODEL_NAME --num-shards=$NUM_GPUS"

SHARD_CMD="python shard.py --model-file=${SHAPED_MODEL} --shard-prefix=${PREFIX} --mode=${MODE} --num-shards=$NUM_GPUS"

COMPARE_CMD="python compare.py --model-file=${ORI_MODEL} --shard-prefix=${PREFIX}"

BENCHMARK="python benchmark.py --model-file=${ORI_MODEL} --seq-len=$SEQ_LEN --batch=$BATCH --loop-cnt=20"

SHARD_BENCHMARK="python benchmark.py --shard-prefix=${PREFIX} --seq-len=$SEQ_LEN --batch=$BATCH --loop-cnt=400"

#PROF="nsys profile -o ${MODEL_NAME}-shard4 -f true --trace=cuda,nvtx,cublas,cudnn "

# working piplines

#$GETMODEL
#$GETGPT2

#$INFER_SHAPE
#$PROCESS_SHAPE
#
#$SHARD_CMD
#
#$PREPARE_INPUTS
#$MPI $COMPARE_CMD


#$PROF $BENCHMARK
$PROF $MPI $SHARD_BENCHMARK
