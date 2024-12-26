#!/bin/bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

# Please run this script under conda or virtual environment with Python 3.10, 3.11 or 3.12.
#   bash benchmark_flux.sh <install_dir> <onnx_dir>

# Installation directory (default: $HOME)
install_dir="${1:-$HOME}"

# Root directory for the onnx models
onnx_dir="${2:-onnx_models}"

# Which GPU to use
export CUDA_VISIBLE_DEVICES=0

# Function to install CUDA 12.6
install_cuda_12()
{
    pushd $install_dir
    wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run
    sh cuda_12.6.2_560.35.03_linux.run --toolkit --toolkitpath=$install_dir/cuda12.6 --silent --override --no-man-page

    export PATH="$install_dir/cuda12.6/bin:$PATH"
    export LD_LIBRARY_PATH="$install_dir/cuda12.6/lib64:$LD_LIBRARY_PATH"
    popd
}

# Function to install cuDNN 9.6
install_cudnn_9() {
    pushd "$install_dir"
    wget  -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.6.0.74_cuda12-archive.tar.xz
    mkdir -p "$install_dir/cudnn9.6"
    tar -Jxvf cudnn-linux-x86_64-9.6.0.74_cuda12-archive.tar.xz  -C "$install_dir/cudnn9.6"--strip=1
    export LD_LIBRARY_PATH="$install_dir/cudnn9.5/lib:$LD_LIBRARY_PATH"
    popd
}

# Install optimum from source before 1.24 is released
install_optimum() {
    pushd "$install_dir"
    optimum_dir="$install_dir/optimum"
    if [ ! -d "$optimum_dir" ]; then
        git clone https://github.com/huggingface/optimum
    fi
    cd "$sam2_dir"
    pip show optimum > /dev/null 2>&1 || pip install -e .
    popd
}

# Install onnxruntime-gpu from source before 1.21 is released
install_onnxruntime() {
    pushd "$install_dir"
    if ! [ -d onnxruntime ]; then
        git clone https://github.com/microsoft/onnxruntime
    fi
    cd onnxruntime
    CUDA_ARCH=$(python3 -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')")
    if [ -n "$CUDA_ARCH" ]; then
        pip install --upgrade pip cmake psutil setuptools wheel packaging ninja numpy==2.2
        sh build.sh --config Release --build_dir build/cuda12 --build_shared_lib --parallel \
                --use_cuda --cuda_version 12.6 --cuda_home $install_dir/cuda12.6 \
                --cudnn_home $install_dir/cudnn9.6 \
                --build_wheel --skip_tests \
                --cmake_generator Ninja \
                --compile_no_warning_as_error \
                --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH
        pip install build/cuda12/Release/dist/onnxruntime_gpu-*-linux_x86_64.whl
    else
        echo "No CUDA device found."
        exit 1
    fi
    popd
}

# Install GPU dependencies
install_gpu() {
    [ ! -d "$install_dir/cuda12.6" ] && install_cuda_12
    [ ! -d "$install_dir/cudnn9.6" ] && install_cudnn_9
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

    pip install diffusers==0.31.0 transformers==4.46.3 onnx==1.17.0 protobuf==5.29.2

    install_onnxruntime

    install_optimum
}

run_benchmark() {
    local model=$1
    local dir=$2
    local version=$3
    local steps=$4
    local batch=$5

    mkdir -p $dir
    [ ! -d "$dir/fp32" ] &&  optimum-cli export onnx --model $model $dir/fp32 --opset 15 --task text-to-image
    [ ! -d "$dir/fp16_fp32" ] && python optimize_pipeline.py -i $dir/fp32 -o $dir/fp16_fp32 --float16
    [ ! -d "$dir/fp16_bf16" ] && python optimize_pipeline.py -i $dir/fp32 -o $dir/fp16_bf16 --float16 --bfloat16
    python benchmark.py -e optimum --height 1024 --width 1024 --steps $steps -b $batch -v $version -p $dir/fp16_fp32
    python benchmark.py -e optimum --height 1024 --width 1024 --steps $steps -b $batch -v $version -p $dir/fp16_bf16
    python benchmark.py -e torch --height 1024 --width 1024 --steps $steps -b $batch -v $version
    python benchmark.py -e torch --height 1024 --width 1024 --steps $steps -b $batch -v $version --enable_torch_compile
}

install_gpu

mkdir -p $root_dir

run_benchmark black-forest-labs/FLUX.1-schnell ${root_dir}/flux1_schnell Flux.1S 4 1 > $root_dir/flux1_schnell_s4_b1.log
run_benchmark black-forest-labs/FLUX.1-dev ${root_dir}/flux1_dev Flux.1D 50 1 > $root_dir/flux1_dev_s50_b1.log

run_benchmark stabilityai/stable-diffusion-3.5-large ${root_dir}/sd3.5_large 3.5L 50 1 > $root_dir/sd3.5_large_s50_b1.log
run_benchmark stabilityai/stable-diffusion-3.5-medium ${root_dir}/sd3.5_medium 3.5M 50 1 > $root_dir/sd3.5_medium_s50_b1.log
