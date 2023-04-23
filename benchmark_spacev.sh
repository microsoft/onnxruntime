#!/bin/bash

set -ex

THIS_DIR=$(dirname $(realpath $0))

export PYTHONPATH=${THIS_DIR}/build_rocm/Release/build/lib

bs=64
sl=512
model_dir=${THIS_DIR}/../benchmark_suite/payload/inference/bench_models
m=${model_dir}/spacev_fp16_rocm_no_attention_fusion.onnx

test_times=400

export ROC_USE_FGS_KERNARG=0

ROCR_VISIBLE_DEVICES=0 \
/usr/bin/numactl \
 --physcpubind=0-23 \
 --preferred=0 \
python -m onnxruntime.transformers.bert_perf_test \
  --model $m --batch_size $bs --sequence_length $sl \
  --intra_op_num_threads 24 --use_gpu --provider rocm --opt_level=99 --use_io_binding --log_severity=0 --test_times ${test_times} \
  --input_ids_name input_ids --input_mask_name attn_mask \
# & \
# ROCR_VISIBLE_DEVICES=1 \
# /usr/bin/numactl \
#  --physcpubind=0-23 \
#  --preferred=0 \
# python -m onnxruntime.transformers.bert_perf_test \
#   --model $m --batch_size $bs --sequence_length $sl \
#   --intra_op_num_threads 24 --use_gpu --provider rocm --opt_level=99 --use_io_binding --log_severity=0 --test_times ${test_times} \
#   --input_ids_name input_ids --input_mask_name attn_mask
