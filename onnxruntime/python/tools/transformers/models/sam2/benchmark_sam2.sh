#!/bin/bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

# Please refer to README.md for the prerequisites and usage of this script.
#   bash benchmark_sam2.sh <install_dir> <cpu|gpu> [profiling]

python="$CONDA_PREFIX/bin/python3"

# Directory of the script and ONNX models
dir="$(cd "$(dirname "$0")" && pwd)"
onnx_dir="$dir/sam2_onnx_models"

# Installation directory (default: $HOME)
install_dir="${1:-$HOME}"

if [ ! -d "$install_dir" ]; then
    echo "Error: install_dir '$install_dir' does not exist."
    exit 1
fi

# SAM2 code directory and model to benchmark
sam2_dir="$install_dir/segment-anything-2"
model="sam2_hiera_large"

# Default to GPU, switch to CPU if specified
cpu_or_gpu="${2:-gpu}"
if [ "$cpu_or_gpu" != "gpu" ] && [ "$cpu_or_gpu" != "cpu" ]; then
    echo "Invalid option: $2. Please specify 'cpu' or 'gpu'."
    exit 1
fi

echo "install_dir: $install_dir"
echo "cpu_or_gpu: $cpu_or_gpu"

install_cuda_12()
{
    pushd $install_dir
    wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run
    sh cuda_12.6.2_560.35.03_linux.run --toolkit --toolkitpath=$install_dir/cuda12.6 --silent --override --no-man-page

    export PATH="$install_dir/cuda12.6/bin:$PATH"
    export LD_LIBRARY_PATH="$install_dir/cuda12.6/lib64:$LD_LIBRARY_PATH"
    popd
}

# Function to install cuDNN 9.4
install_cudnn_9() {
    pushd "$install_dir"
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.5.0.50_cuda12-archive.tar.xz
    mkdir -p "$install_dir/cudnn9.5"
    tar -Jxvf cudnn-linux-x86_64-9.5.0.50_cuda12-archive.tar.xz -C "$install_dir/cudnn9.5" --strip=1
    export LD_LIBRARY_PATH="$install_dir/cudnn9.5/lib:$LD_LIBRARY_PATH"
    popd
}

# Install GPU dependencies
install_gpu() {
    [ ! -d "$install_dir/cuda12.6" ] && install_cuda_12
    [ ! -d "$install_dir/cudnn9.5" ] && install_cudnn_9

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install onnxruntime-gpu onnx opencv-python matplotlib
}

# Install CPU dependencies
install_cpu() {
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install onnxruntime onnx opencv-python matplotlib
}

# Clone and install SAM2 if not already installed
install_sam2() {
    pushd "$install_dir"
    if [ ! -d "$sam2_dir" ]; then
        git clone https://github.com/facebookresearch/segment-anything-2.git
    fi
    cd "$sam2_dir"
    pip show SAM-2 > /dev/null 2>&1 || pip install -e .
    [ ! -f checkpoints/sam2_hiera_large.pt ] && (cd checkpoints && sh ./download_ckpts.sh)
    popd
}

# Download test image if not available
download_test_image() {
    [ ! -f truck.jpg ] && curl -sO https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/notebooks/images/truck.jpg
}

run_cpu_benchmark() {
    local repeats="$1"
    $python convert_to_onnx.py --sam2_dir "$sam2_dir" --optimize --demo

    for component in image_encoder image_decoder; do
        $python benchmark_sam2.py --model_type "$model" --engine torch --sam2_dir "$sam2_dir" --repeats "$repeats" --dtype fp32 --component "$component"

        # Run ONNX Runtime on exported model (not optimized)
        $python benchmark_sam2.py --model_type "$model" --engine ort --sam2_dir "$sam2_dir" --repeats "$repeats" --onnx_path "${onnx_dir}/${model}_${component}.onnx" --dtype fp32 --component "$component"

        # Run ONNX Runtime on optimized model
        $python benchmark_sam2.py --model_type "$model" --engine ort --sam2_dir "$sam2_dir" --repeats "$repeats" --onnx_path "${onnx_dir}/${model}_${component}_fp32_cpu.onnx" --dtype fp32 --component "$component"
    done
}

run_gpu_benchmark() {
    local repeats="$1"
    $python convert_to_onnx.py --sam2_dir "$sam2_dir" --optimize --use_gpu --dtype fp32
    $python convert_to_onnx.py --sam2_dir "$sam2_dir" --optimize --use_gpu --dtype fp16 --demo

    for component in image_encoder image_decoder; do
        for dtype in bf16 fp32 fp16; do
            $python benchmark_sam2.py --model_type "$model" --engine torch --sam2_dir "$sam2_dir" --repeats "$repeats" --use_gpu --dtype $dtype --component "$component"
        done
    done

    component="image_encoder"
    for dtype in fp32 fp16; do
        #TODO: --prefer_nhwc does not help with performance
        $python benchmark_sam2.py --model_type "$model" --engine ort --sam2_dir "$sam2_dir" --repeats "$repeats" --use_gpu --dtype $dtype --component "$component" --onnx_path "${onnx_dir}/${model}_${component}_${dtype}_gpu.onnx" --use_cuda_graph
    done

    component="image_decoder"
    for dtype in fp32 fp16; do
        # TODO: decoder does not work with cuda graph
        $python benchmark_sam2.py --model_type "$model" --engine ort --sam2_dir "$sam2_dir" --repeats "$repeats" --use_gpu --dtype $dtype --component "$component" --onnx_path "${onnx_dir}/${model}_${component}_${dtype}_gpu.onnx"
    done
}

run_torch_compile_gpu_benchmark() {
    local repeats="$1"

    # Test different torch compile modes on image encoder
    for torch_compile_mode in none max-autotune reduce-overhead max-autotune-no-cudagraphs
    do
        $python benchmark_sam2.py --model_type $model --engine torch --sam2_dir "$sam2_dir" --repeats "$repeats" --use_gpu --dtype fp16 --component image_encoder --torch_compile_mode $torch_compile_mode
    done
}


# Main script
run_benchmarks() {
    if [ ! -v CONDA_PREFIX ]; then
        echo "Please activate conda environment before running this script."
        exit 1
    fi

    # Install dependencies
    [ "$cpu_or_gpu" = "gpu" ] && install_gpu || install_cpu
    install_sam2
    download_test_image

    # Run benchmarks
    output_csv="sam2_${cpu_or_gpu}.csv"
    if [ ! -f "$output_csv" ]; then
        echo "Running $cpu_or_gpu benchmark..."
        if [ "$cpu_or_gpu" = "gpu" ]; then
            run_gpu_benchmark 1000
            run_torch_compile_gpu_benchmark 1000
        else
            run_cpu_benchmark 100
        fi
        cat benchmark*.csv > combined_csv
        awk '!x[$0]++' combined_csv > "$output_csv"
        rm combined_csv
        echo "Benchmark results saved in $output_csv"
    else
        echo "$output_csv already exists, skipping benchmark..."
    fi
}

run_benchmarks

#--------------------------------------------------------------------------
# Below are for profiling
#--------------------------------------------------------------------------

# Build onnxruntime-gpu from source for profiling
build_onnxruntime_gpu_for_profiling() {
    pushd "$install_dir"
    if ! [ -d onnxruntime ]; then
        git clone https://github.com/microsoft/onnxruntime
    fi
    cd onnxruntime
    CUDA_ARCH=$(python3 -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')")
    if [ -n "$CUDA_ARCH" ]; then
        pip install --upgrade pip cmake psutil setuptools wheel packaging ninja numpy==1.26.4
        sh build.sh --config Release --build_dir build/cuda12 --build_shared_lib --parallel \
                --use_cuda --cuda_version 12.6 --cuda_home $install_dir/cuda12.6 \
                --cudnn_home $install_dir/cudnn9.5 \
                --build_wheel --skip_tests \
                --cmake_generator Ninja \
                --compile_no_warning_as_error \
                --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH \
                --cmake_extra_defines onnxruntime_ENABLE_NVTX_PROFILE=ON \
                --enable_cuda_line_info

        pip install build/cuda12/Release/dist/onnxruntime_gpu-*-linux_x86_64.whl numpy==1.26.4
    else
        echo "No CUDA device found."
        exit 1
    fi
    popd
}

# Run profiling with NVTX.
run_nvtx_profile()
{
    pip install nvtx cuda-python==12.6.0

    # Only trace one device to avoid huge output file size.
    device_id=0
    envs="CUDA_VISIBLE_DEVICES=$device_id,ORT_ENABLE_CUDNN_FLASH_ATTENTION=1,LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    cuda_graph_trace=node
    for engine in ort torch; do
        for component in image_encoder image_decoder; do
            sudo $install_dir/cuda12.6/bin/nsys profile --capture-range=nvtx --nvtx-capture='one_run' \
                --gpu-metrics-device $device_id --force-overwrite true \
                --sample process-tree --backtrace fp --stats true \
                -t cuda,cudnn,cublas,osrt,nvtx --cuda-memory-usage true --cudabacktrace all \
                --cuda-graph-trace $cuda_graph_trace \
                -e $envs,NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
                -o sam2_fp16_profile_${component}_${engine}_${cpu_or_gpu} \
                $python benchmark_sam2.py --model_type $model --engine $engine \
                                          --sam2_dir  $sam2_dir --warm_up 1 --repeats 0 \
                                          --onnx_path ${onnx_dir}/${model}_${component}_fp16_gpu.onnx \
                                          --component $component \
                                          --use_gpu --dtype fp16 --enable_nvtx_profile
        done
    done
}

# Run profiling with PyTorch
run_torch_profile() {
    for component in image_encoder image_decoder; do
        $python benchmark_sam2.py --model_type $model --engine torch \
                                  --sam2_dir  $sam2_dir --warm_up 1 --repeats 0 \
                                  --component $component \
                                  --use_gpu --dtype fp16 --enable_torch_profile
    done
}

run_profilings() {
    build_onnxruntime_gpu_for_profiling

    rm -f *.nsys-rep *.sqlite
    run_nvtx_profile

    run_torch_profile
}

profiling="${3:-false}"
if [ "$profiling" = "true" ] &&  [ "$cpu_or_gpu" = "gpu" ]; then
    run_profilings
fi
