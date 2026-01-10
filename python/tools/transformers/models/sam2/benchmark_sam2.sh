#!/bin/bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

# Please refer to README.md for the prerequisites and usage of this script.
#   bash benchmark_sam2.sh <install_dir> <cpu|gpu> [profiling] [benchmarking] [nightly] [dynamo]
# Note that dynamo need onnxruntime 1.21 or later, or nightly build.
# Example:
#   bash benchmark_sam2.sh $HOME gpu true true true false

install_dir="${1:-$HOME}"
cpu_or_gpu="${2:-gpu}"
profiling="${3:-false}"
benchmarking="${4:-true}"
nightly="${5:-false}"
dynamo="${6:-false}"

python="$CONDA_PREFIX/bin/python3"

# Directory of the script and ONNX models
dir="$(cd "$(dirname "$0")" && pwd)"
onnx_dir="$dir/sam2_onnx_models"

if [ ! -d "$install_dir" ]; then
    echo "Error: install_dir '$install_dir' does not exist."
    exit 1
fi

# SAM2 code directory and model to benchmark
sam2_dir="$install_dir/segment-anything-2"
model="sam2_hiera_large"

# Default to GPU, switch to CPU if specified
if [ "$cpu_or_gpu" != "gpu" ] && [ "$cpu_or_gpu" != "cpu" ]; then
    echo "Invalid option: $2. Please specify 'cpu' or 'gpu'."
    exit 1
fi

echo "install_dir: $install_dir"
echo "cpu_or_gpu: $cpu_or_gpu"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Ensure necessary tools are installed
if ! command_exists wget; then
    echo "wget is not installed. Please install it and try again."
    exit 1
fi

if ! command_exists git; then
    echo "git is not installed. Please install it and try again."
    exit 1
fi

if ! command_exists pip; then
    echo "pip is not installed. Please install it and try again."
    exit 1
fi

cuda_version=12.6
cudnn_version=9.5

# Install CUDA 12.6
install_cuda_12() {
    if ! [ -d "$install_dir/cuda${cuda_version}" ]; then
        pushd "$install_dir" || exit
        wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run
        sh cuda_12.6.2_560.35.03_linux.run --toolkit --toolkitpath="$install_dir/cuda${cuda_version}" --silent --override --no-man-page
        popd || exit
    fi
    export PATH="$install_dir/cuda${cuda_version}/bin:$PATH"
    export LD_LIBRARY_PATH="$install_dir/cuda${cuda_version}/lib64:$LD_LIBRARY_PATH"
}

# Install cuDNN 9.5
install_cudnn_9() {
    if ! [ -d "$install_dir/cudnn${cudnn_version}" ]; then
        pushd "$install_dir" || exit
        wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.5.0.50_cuda12-archive.tar.xz
        mkdir -p "$install_dir/cudnn${cudnn_version}"
        tar -Jxvf cudnn-linux-x86_64-9.5.0.50_cuda12-archive.tar.xz -C "$install_dir/cudnn${cudnn_version}" --strip=1
        popd || exit
    fi
    export LD_LIBRARY_PATH="$install_dir/cudnn${cudnn_version}/lib:$LD_LIBRARY_PATH"
}

install_ort() {
    local ort="$1"
    pip uninstall onnxruntime onnxruntime-gpu -y

    if [ "$nightly" = "true" ]; then
        pip install flatbuffers numpy packaging protobuf sympy
        pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ "$ort"
    else
        pip install "$ort"
    fi

    pip install onnx onnxscript opencv-python matplotlib
}

# Install GPU dependencies
install_gpu() {
    install_cuda_12
    install_cudnn_9
    echo "PATH: $PATH"
    echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

    # The dynamo export need torch 2.6.0 or later. Use the latest one.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --upgrade

    install_ort "onnxruntime-gpu"
}

# Install CPU dependencies
install_cpu() {
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    install_ort "onnxruntime"
}

# Clone and install SAM2 if not already installed
install_sam2() {
    pushd "$install_dir" || exit
    if [ ! -d "$sam2_dir" ]; then
        git clone https://github.com/facebookresearch/segment-anything-2.git
    fi
    cd "$sam2_dir" || exit
    pip show SAM-2 > /dev/null 2>&1 || pip install -e .
    [ ! -f checkpoints/sam2_hiera_large.pt ] && (cd checkpoints && sh ./download_ckpts.sh)
    popd || exit
}

# Download test image if not available
download_test_image() {
    [ ! -f truck.jpg ] && curl -sO https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/notebooks/images/truck.jpg
}

run_cpu_benchmark() {
    local repeats="$1"

    if [ "$dynamo" = "true" ]; then
        $python convert_to_onnx.py --sam2_dir "$sam2_dir" --optimize --demo --dynamo
    else
        $python convert_to_onnx.py --sam2_dir "$sam2_dir" --optimize --demo
    fi

    for component in image_encoder image_decoder; do
        $python benchmark_sam2.py --model_type "$model" --engine torch --sam2_dir "$sam2_dir" --repeats "$repeats" --dtype fp32 --component "$component"

        # Run ONNX Runtime on exported model (not optimized)
        $python benchmark_sam2.py --model_type "$model" --engine ort --sam2_dir "$sam2_dir" --repeats "$repeats" --onnx_path "${onnx_dir}/${model}_${component}.onnx" --dtype fp32 --component "$component"

        # Run ONNX Runtime on optimized model
        $python benchmark_sam2.py --model_type "$model" --engine ort --sam2_dir "$sam2_dir" --repeats "$repeats" --onnx_path "${onnx_dir}/${model}_${component}_fp32_cpu.onnx" --dtype fp32 --component "$component"
    done
}

run_ort_gpu_benchmark() {
    local repeats="$1"

    if [ "$dynamo" = "true" ]; then
        $python convert_to_onnx.py --sam2_dir "$sam2_dir" --optimize --use_gpu --dtype fp32 --dynamo
        $python convert_to_onnx.py --sam2_dir "$sam2_dir" --optimize --use_gpu --dtype fp16 --demo --dynamo
    else
        $python convert_to_onnx.py --sam2_dir "$sam2_dir" --optimize --use_gpu --dtype fp32
        $python convert_to_onnx.py --sam2_dir "$sam2_dir" --optimize --use_gpu --dtype fp16 --demo
    fi

    component="image_encoder"
    for dtype in fp32 fp16; do
        $python benchmark_sam2.py --model_type "$model" --engine ort --sam2_dir "$sam2_dir" --repeats "$repeats" --use_gpu --dtype "$dtype" --component "$component" --onnx_path "${onnx_dir}/${model}_${component}_${dtype}_gpu.onnx" --use_cuda_graph
    done
    # Test prefer_nhwc.
    $python benchmark_sam2.py --model_type "$model" --engine ort --sam2_dir "$sam2_dir" --repeats "$repeats" --use_gpu --dtype fp16 --component "$component" --onnx_path "${onnx_dir}/${model}_${component}_${dtype}_gpu.onnx" --use_cuda_graph --prefer_nhwc

    component="image_decoder"
    for dtype in fp32 fp16; do
        # TODO: decoder does not work with cuda graph
        $python benchmark_sam2.py --model_type "$model" --engine ort --sam2_dir "$sam2_dir" --repeats "$repeats" --use_gpu --dtype "$dtype" --component "$component" --onnx_path "${onnx_dir}/${model}_${component}_${dtype}_gpu.onnx"
    done
    # Test prefer_nhwc.
    $python benchmark_sam2.py --model_type "$model" --engine ort --sam2_dir "$sam2_dir" --repeats "$repeats" --use_gpu --dtype fp16 --component "$component" --onnx_path "${onnx_dir}/${model}_${component}_${dtype}_gpu.onnx" --prefer_nhwc
}

run_torch_gpu_benchmark() {
    local repeats="$1"

    # Test PyTorch eager mode.
    for component in image_encoder image_decoder; do
        for dtype in bf16 fp32 fp16; do
            $python benchmark_sam2.py --model_type "$model" --engine torch --sam2_dir "$sam2_dir" --repeats "$repeats" --use_gpu --dtype "$dtype" --component "$component"
        done
    done

    # Test different torch compile modes on image encoder
    for torch_compile_mode in none max-autotune reduce-overhead max-autotune-no-cudagraphs
    do
        $python benchmark_sam2.py --model_type "$model" --engine torch --sam2_dir "$sam2_dir" --repeats "$repeats" --use_gpu --dtype fp16 --component image_encoder --torch_compile_mode $torch_compile_mode
    done
}

install_all() {
    if [ "$cpu_or_gpu" = "gpu" ]; then
        install_gpu
    else
        install_cpu
    fi
    install_sam2
    download_test_image
}

run_benchmarks() {
    suffix=$(date +"%Y_%m_%d_%H_%M_%S")
    [ "$dynamo" = "true" ] && suffix="${suffix}_dynamo"
    output_csv="sam2_${cpu_or_gpu}_${suffix}.csv"
    if [ ! -f "$output_csv" ]; then
        echo "Running $cpu_or_gpu benchmark..."
        if [ "$cpu_or_gpu" = "gpu" ]; then
            run_ort_gpu_benchmark 1000
            run_torch_gpu_benchmark 1000
        else
            run_cpu_benchmark 100
        fi
        cat benchmark*.csv > combined_csv
        awk '!x[$0]++' combined_csv > "$output_csv"
        rm benchmark*.csv
        rm combined_csv
        echo "Benchmark results saved in $output_csv"
    else
        echo "$output_csv already exists, skipping benchmark..."
    fi
}

if [ ! -v CONDA_PREFIX ]; then
    echo "Please activate conda environment before running this script."
    exit 1
fi

install_all

if [ "$benchmarking" = "true" ]; then
    run_benchmarks
fi

#--------------------------------------------------------------------------
# Below are for profiling
#--------------------------------------------------------------------------

# Build onnxruntime-gpu from source for profiling
build_onnxruntime_gpu_for_profiling() {
    pushd "$install_dir" || exit
    if ! [ -d onnxruntime ]; then
        git clone https://github.com/microsoft/onnxruntime
    fi
    cd onnxruntime || exit
    pip install --upgrade pip cmake psutil setuptools wheel packaging ninja numpy
    build_dir=build/cuda${cuda_version}
    rm -rf ${build_dir}/Release/dist
    sh build.sh --config Release --build_dir "${build_dir}" --build_shared_lib --parallel \
            --use_cuda --cuda_version ${cuda_version} --cuda_home "$install_dir/cuda${cuda_version}" \
            --cudnn_home "$install_dir/cudnn${cudnn_version}" \
            --build_wheel --skip_tests \
            --cmake_generator Ninja \
            --compile_no_warning_as_error \
            --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=native \
            --cmake_extra_defines onnxruntime_ENABLE_NVTX_PROFILE=ON \
            --enable_cuda_line_info
    pip uninstall onnxruntime-gpu -y
    pip install "${build_dir}/Release/dist/onnxruntime_gpu-*-linux_x86_64.whl"
    popd || exit
}

# Run profiling with NVTX.
run_nvtx_profile() {
    local engine="$1"
    # Only trace one device to avoid huge output file size.
    device_id=0
    envs="CUDA_VISIBLE_DEVICES=$device_id,ORT_ENABLE_CUDNN_FLASH_ATTENTION=1,LD_LIBRARY_PATH=$LD_LIBRARY_PATH,TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1"
    cuda_graph_trace=node
    for component in image_encoder image_decoder; do
        sudo "$install_dir/cuda${cuda_version}/bin/nsys" profile --capture-range=nvtx --nvtx-capture='one_run' \
            --gpu-metrics-devices $device_id --force-overwrite true \
            --sample process-tree --backtrace fp --stats true \
            -t cuda,cudnn,cublas,osrt,nvtx --cuda-memory-usage true --cudabacktrace all \
            --cuda-graph-trace "$cuda_graph_trace" \
            -e "$envs,NSYS_NVTX_PROFILER_REGISTER_ONLY=0" \
            -o "sam2_fp16_profile_${component}_${engine}_${cpu_or_gpu}" \
            $python benchmark_sam2.py --model_type "$model" --engine "$engine" \
                                      --sam2_dir "$sam2_dir" --warm_up 1 --repeats 0 \
                                      --onnx_path "${onnx_dir}/${model}_${component}_fp16_gpu.onnx" \
                                      --component "$component" \
                                      --use_gpu --dtype fp16 --enable_nvtx_profile
    done
}

run_ort_profile() {
    export ORT_ENABLE_CUDNN_FLASH_ATTENTION=1
    rm -f onnxruntime_*.json
    for component in image_encoder image_decoder; do
        $python benchmark_sam2.py --model_type "$model" --engine ort \
                                  --sam2_dir  "$sam2_dir" --warm_up 1 --repeats 0 \
                                  --onnx_path "${onnx_dir}/${model}_${component}_fp16_gpu.onnx" \
                                  --component "$component" \
                                  --use_gpu --dtype fp16 --enable_ort_profile
        mv onnxruntime_profile*.json onnxruntime_$component.json
    done
}

# Run profiling with PyTorch
run_torch_profile() {
    # Enable logging might could help get the code of compiled kernels. You can turn it off to reduce overhead.
    export TORCH_LOGS="+inductor,+output_code"
    export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
    component=image_encoder
    $python benchmark_sam2.py --model_type "$model" --engine torch \
                              --sam2_dir "$sam2_dir" --warm_up 1 --repeats 0 \
                              --component "$component" \
                              --torch_compile_mode max-autotune \
                              --use_gpu --dtype fp16 --enable_torch_profile > "torch_${component}_compiled_code.txt"

    component=image_decoder
    $python benchmark_sam2.py --model_type "$model" --engine torch \
                              --sam2_dir "$sam2_dir" --warm_up 1 --repeats 0 \
                              --component "$component" \
                              --torch_compile_mode none \
                              --use_gpu --dtype fp16 --enable_torch_profile
}

run_nvtx_profilings() {
    build_onnxruntime_gpu_for_profiling
    rm -f *.nsys-rep *.sqlite
    run_nvtx_profile ort
    run_nvtx_profile torch
}

run_profilings() {
    pip install nvtx cuda-python==${cuda_version}.0
    run_ort_profile
    run_torch_profile

    # NVTX profiling need to build onnxruntime-gpu from source so it is put as the last step.
    run_nvtx_profilings
}

if [ "$profiling" = "true" ] &&  [ "$cpu_or_gpu" = "gpu" ]; then
    run_profilings
fi
