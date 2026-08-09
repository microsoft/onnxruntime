#!/bin/bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

set -euo pipefail

# Script to benchmark Flux models with ONNX and PyTorch
# Usage: bash benchmark_flux.sh <install_dir> <onnx_dir>

# Validate inputs and environment
command -v python3 &>/dev/null || { echo "Python3 is required but not installed."; exit 1; }
command -v wget &>/dev/null || { echo "wget is required but not installed."; exit 1; }

# Input arguments with defaults
install_dir="${1:-$HOME}"
onnx_dir="${2:-onnx_models}"

# GPU settings
export CUDA_VISIBLE_DEVICES=0

# Function to log messages
log() {
    echo -e "\033[1;32m[INFO]\033[0m $1"
}

# Function to install CUDA 12.6
install_cuda_12() {
    log "Installing CUDA 12.6"
    pushd "$install_dir"
    wget -q https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run
    sh cuda_12.6.2_560.35.03_linux.run --toolkit --toolkitpath="$install_dir/cuda12.6" --silent --override --no-man-page
    export PATH="$install_dir/cuda12.6/bin:$PATH"
    export LD_LIBRARY_PATH="$install_dir/cuda12.6/lib64:$LD_LIBRARY_PATH"
    popd
}

# Function to install cuDNN 9.6
install_cudnn_9() {
    log "Installing cuDNN 9.6"
    pushd "$install_dir"
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.6.0.74_cuda12-archive.tar.xz
    mkdir -p "$install_dir/cudnn9.6"
    tar -Jxvf cudnn-linux-x86_64-9.6.0.74_cuda12-archive.tar.xz -C "$install_dir/cudnn9.6" --strip=1
    export LD_LIBRARY_PATH="$install_dir/cudnn9.6/lib:$LD_LIBRARY_PATH"
    popd
}

# Function to install optimum
install_optimum() {
    log "Installing Optimum"
    optimum_dir="$install_dir/optimum"
    if [ ! -d "$optimum_dir" ]; then
        git clone https://github.com/huggingface/optimum "$optimum_dir"
    fi
    pushd "$optimum_dir"
    pip show optimum &>/dev/null || pip install -e .
    popd
}

# Function to build and install ONNX Runtime
install_onnxruntime() {
    log "Building ONNX Runtime"
    pushd "$install_dir"
    if [ ! -d onnxruntime ]; then
        git clone https://github.com/microsoft/onnxruntime
    fi
    pushd onnxruntime
    pip install --upgrade pip cmake psutil setuptools wheel packaging ninja numpy==2.2
    sh build.sh --config Release --build_dir build/cuda12 --parallel \
        --use_cuda --cuda_version 12.6 --cuda_home "$install_dir/cuda12.6" \
        --cudnn_home "$install_dir/cudnn9.6" \
        --build_wheel --skip_tests \
        --cmake_generator Ninja \
        --compile_no_warning_as_error \
        --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF CMAKE_CUDA_ARCHITECTURES=native

    log "Installing ONNX Runtime"
    pip install build/cuda12/Release/dist/onnxruntime_gpu-*-linux_x86_64.whl
    popd
    popd
}

# Function to install GPU dependencies
install_gpu() {
    log "Installing GPU dependencies"
    [ ! -d "$install_dir/cuda12.6" ] && install_cuda_12
    [ ! -d "$install_dir/cudnn9.6" ] && install_cudnn_9
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install diffusers==0.32.0 transformers==4.46.3 onnx==1.17.0 protobuf==5.29.2 py3nvml
    install_onnxruntime
    install_optimum
}

# Function to run benchmarks
run_benchmark() {
    local model=$1
    local dir=$2
    local version=$3
    local steps=$4
    local batch=$5

    log "Running benchmark for model: $model"
    mkdir -p "$dir"
    [ ! -d "$dir/fp32" ] && optimum-cli export onnx --model "$model" "$dir/fp32" --opset 15 --task text-to-image
    [ ! -d "$dir/fp16_fp32" ] && python optimize_pipeline.py -i "$dir/fp32" -o "$dir/fp16_fp32" --float16
    [ ! -d "$dir/fp16_bf16" ] && python optimize_pipeline.py -i "$dir/fp32" -o "$dir/fp16_bf16" --float16 --bfloat16
    python benchmark.py -e optimum --height 1024 --width 1024 --steps "$steps" -b "$batch" -v "$version" -p "$dir/fp16_fp32"
    python benchmark.py -e optimum --height 1024 --width 1024 --steps "$steps" -b "$batch" -v "$version" -p "$dir/fp16_bf16"
    python benchmark.py -e torch --height 1024 --width 1024 --steps "$steps" -b "$batch" -v "$version"
    python benchmark.py -e torch --height 1024 --width 1024 --steps "$steps" -b "$batch" -v "$version" --enable_torch_compile
}

# Main script execution
install_gpu

log "Creating ONNX model directory: $onnx_dir"
mkdir -p "$onnx_dir"

run_benchmark black-forest-labs/FLUX.1-schnell "$onnx_dir/flux1_schnell" Flux.1S 4 1 > "$onnx_dir/flux1_schnell_s4_b1.log"
run_benchmark black-forest-labs/FLUX.1-dev "$onnx_dir/flux1_dev" Flux.1D 50 1 > "$onnx_dir/flux1_dev_s50_b1.log"
run_benchmark stabilityai/stable-diffusion-3.5-large "$onnx_dir/sd3.5_large" 3.5L 50 1 > "$onnx_dir/sd3.5_large_s50_b1.log"
run_benchmark stabilityai/stable-diffusion-3.5-medium "$onnx_dir/sd3.5_medium" 3.5M 50 1 > "$onnx_dir/sd3.5_medium_s50_b1.log"

log "Benchmark completed."
