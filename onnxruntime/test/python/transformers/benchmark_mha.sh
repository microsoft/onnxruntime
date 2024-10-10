#!/bin/sh

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

echo "Benchmark Scaled Dot Product Attention (SDPA) performance on GPU:"

export CUDA_VISIBLE_DEVICES=0
python benchmark_mha.py --use_gpu

echo "Benchmark BERT-Large performance on GPU without attention bias"
python benchmark_mha.py --use_gpu -b 16

echo "Benchmark BERT-Large performance on GPU with attention bias"
python benchmark_mha.py --use_gpu -b 16 -r 1000 --has_attn_bias
python benchmark_mha.py --use_gpu -b 16 -r 1000 --has_attn_bias --broadcast_attn_bias_dim_0
python benchmark_mha.py --use_gpu -b 16 -r 1000 --has_attn_bias --broadcast_attn_bias_dim_0 --broadcast_attn_bias_dim_1

python benchmark_mha.py --use_gpu --use_cuda_graph
python benchmark_mha.py --use_gpu --torch

cat benchmark_mha_gpu_*.csv > mha_gpu_benchmark_results.csv

echo "Benchmark performance on CPU with number of threads:"
MKL_DYNAMIC=FALSE OMP_NUM_THREADS=1 python benchmark_mha.py --torch
MKL_DYNAMIC=FALSE OMP_NUM_THREADS=2 python benchmark_mha.py --torch
MKL_DYNAMIC=FALSE OMP_NUM_THREADS=4 python benchmark_mha.py --torch
MKL_DYNAMIC=FALSE OMP_NUM_THREADS=8 python benchmark_mha.py --torch

python benchmark_mha.py --intra_op_num_threads 1
python benchmark_mha.py --intra_op_num_threads 2
python benchmark_mha.py --intra_op_num_threads 4
python benchmark_mha.py --intra_op_num_threads 8


echo "Benchmark performance on CPU with default threads settings:"
python benchmark_mha.py
ORT_DISABLE_FLASH_ATTENTION=1 python benchmark_mha.py
python benchmark_mha.py --torch

python benchmark_mha.py --causal
python benchmark_mha.py --torch --causal

# Pytorch SDPA does not support causal attention with past state, we only test ORT here.
python benchmark_mha.py --causal --has_past

cat benchmark_mha_cpu_*.csv > mha_cpu_benchmark_results.csv
