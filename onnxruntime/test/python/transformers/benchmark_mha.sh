#!/bin/sh

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Usage: benchmark_mha.sh [gpu|cpu|lean]
task="${1:-gpu}"

# Function to lock GPU clocks and set power limit for a GPU
configure_gpu() {
    local gpu_id=$1

    # Ensure nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null
    then
        echo "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
        exit
    fi

    # Enable Persistence Mode
    sudo nvidia-smi -pm 1 -i $gpu_id

    # Get the maximum clock speeds for graphics and memory.
    nvidia-smi -q -d CLOCK -i ${gpu_id} | grep -A3 "Max Clocks"
    max_graphics_clock=$(nvidia-smi -q -d CLOCK -i ${gpu_id} | grep -A1 "Max Clocks" | grep "Graphics" | awk '{print $3}')
    max_memory_clock=$(nvidia-smi -q -d CLOCK -i ${gpu_id} | grep -A3 "Max Clocks" | grep "Memory" | awk '{print $3}')

    # Lock the GPU clocks to maximum frequencies
    sudo nvidia-smi -i $gpu_id --lock-gpu-clocks=$max_graphics_clock,$max_graphics_clock
    sudo nvidia-smi -i $gpu_id --lock-memory-clocks=$max_memory_clock,$max_memory_clock

    nvidia-smi --query-gpu=clocks.gr,clocks.sm,clocks.mem --format=csv
    echo "GPU $gpu_id clocks locked to $max_graphics_clock MHz (graphics) and $max_memory_clock MHz (memory)"

    # Set Power Limit to maximum
    power_limit=$(nvidia-smi --query-gpu=power.limit -i 0 --format=csv | grep "0" | awk '{print $1}')
    power_limit=${power_limit%.*}
    sudo nvidia-smi -pl $power_limit -i $gpu_id

    export CUDA_VISIBLE_DEVICES=$gpu_id
}

run_gpu_benchmarks() {
    echo "Benchmark Scaled Dot Product Attention (SDPA) performance on GPU:"

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
}

run_lean_benchmarks() {
    echo "Benchmark long context decoding performance on GPU"
    for b in 1 4 16; do
        for s in 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536; do
            python benchmark_mha.py --use_gpu --causal -b $b -s 1 -p $s -n 16 -d 64 -r 1000 --csv_filename_prefix benchmark_lean
            python benchmark_mha.py --use_gpu --causal -b $b -s 1 -p $s -n 32 -d 128 -r 1000 --csv_filename_prefix benchmark_lean
        done
    done
    cat benchmark_lean_*.csv > lean_benchmark_results.csv
}

run_cpu_benchmarks() {
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
}

[ "$task" != "cpu" ] && configure_gpu 0

[ "$task" == "gpu" ] && run_gpu_benchmarks

[ "$task" == "cpu" ] && run_cpu_benchmarks

[ "$task" == "lean" ] && run_lean_benchmarks
