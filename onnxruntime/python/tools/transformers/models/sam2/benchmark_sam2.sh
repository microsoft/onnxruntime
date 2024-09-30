#!/bin/sh
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Here assumes that we are using conda (Anaconda/Miniconda/Miniforge) environment.
# For example, you can create a new conda environment like the following before running this script:
#    conda create -n sam2_gpu python=3.11 -y
#    conda activate sam2_gpu
#    sh benchmark_sam2.sh $HOME gpu
# Or create a new conda environment for CPU benchmark:
#    conda create -n sam2_cpu python=3.11 -y
#    conda activate sam2_cpu
#    sh benchmark_sam2.sh $HOME cpu

python=$CONDA_PREFIX/bin/python3

# Directory of the script
dir="$( cd "$( dirname "$0" )" && pwd )"

# Directory of the onnx models
onnx_dir=$dir/sam2_onnx_models

# Directory to install CUDA, cuDNN, and git clone sam2 or onnxruntime source code.
install_dir=$HOME
if [ $# -ge 1 ]; then
    install_dir=$1
fi

if ! [ -d $install_dir ]; then
    echo "install_dir: $install_dir does not exist."
    exit 1
fi

# Directory of the sam2 code by "git clone https://github.com/facebookresearch/segment-anything-2"
sam2_dir=$install_dir/segment-anything-2

# model name to benchmark
model=sam2_hiera_large

# Default to use GPU if available.
cpu_or_gpu="gpu"
if [ $# -ge 2 ] && ([ "$2" = "gpu" ] || [ "$2" = "cpu" ]); then
    cpu_or_gpu=$2
fi

echo "install_dir: $install_dir"
echo "cpu_or_gpu: $cpu_or_gpu"

install_cuda_12()
{
    pushd $install_dir
    wget https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/cuda_12.5.1_555.42.06_linux.run
    sh cuda_12.5.1_555.42.06_linux.run --toolkit --toolkitpath=$install_dir/cuda12.5 --silent --override --no-man-page

    export PATH="$install_dir/cuda12.5/bin:$PATH"
    export LD_LIBRARY_PATH="$install_dir/cuda12.5/lib64:$LD_LIBRARY_PATH"
    popd
}

install_cudnn_9()
{
    pushd $install_dir
    wget  https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.4.0.58_cuda12-archive.tar.xz
    mkdir $install_dir/cudnn9.4
    tar -Jxvf cudnn-linux-x86_64-9.4.0.58_cuda12-archive.tar.xz -C $install_dir/cudnn9.4 --strip=1  --no-overwrite-dir

    export LD_LIBRARY_PATH="$install_dir/cudnn9.4/lib:$LD_LIBRARY_PATH"
    popd
}

install_gpu()
{
    if ! [ -d $install_dir/cuda12.5 ]; then
        install_cuda_12
    fi

    if ! [ -d $install_dir/cudnn9.4 ]; then
        install_cudnn_9
    fi

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install onnxruntime-gpu onnx opencv-python matplotlib
}

install_cpu()
{
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install onnxruntime onnx opencv-python matplotlib
}

install_sam2()
{
    pushd $install_dir

    if ! [ -d $install_dir/segment-anything-2 ]; then
        git clone https://github.com/facebookresearch/segment-anything-2.git
    fi

    cd segment-anything-2

    if pip show SAM-2 > /dev/null 2>&1; then
        echo "SAM-2 is already installed."
    else
        pip install -e .
    fi

    if ! [ -f checkpoints/sam2_hiera_large.pt ]; then
        echo "Downloading checkpoints..."
        cd checkpoints
        sh ./download_ckpts.sh
    fi

    popd
}

download_test_image()
{
    if ! [ -f truck.jpg ]; then
        curl https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/notebooks/images/truck.jpg > truck.jpg
    fi
}

run_cpu()
{
    repeats=$1

    $python convert_to_onnx.py  --sam2_dir $sam2_dir --optimize --demo

    echo "Benchmarking SAM2 model $model image encoder for PyTorch ..."
    $python benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --dtype fp32
    $python benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --dtype fp16

    echo "Benchmarking SAM2 model $model image encoder for PyTorch ..."
    $python benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --dtype fp32 --component image_decoder
    $python benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --dtype fp16 --component image_decoder

    echo "Benchmarking SAM2 model $model image encoder for ORT ..."
    $python benchmark_sam2.py --model_type $model --engine ort --sam2_dir $sam2_dir --repeats $repeats --onnx_path ${onnx_dir}/${model}_image_encoder.onnx --dtype fp32
    $python benchmark_sam2.py --model_type $model --engine ort --sam2_dir $sam2_dir --repeats $repeats --onnx_path ${onnx_dir}/${model}_image_encoder_fp32_cpu.onnx --dtype fp32

    echo "Benchmarking SAM2 model $model image decoder for ORT ..."
    $python benchmark_sam2.py --model_type $model --engine ort --sam2_dir $sam2_dir --repeats $repeats --onnx_path ${onnx_dir}/${model}_image_decoder.onnx --component image_decoder
    $python benchmark_sam2.py --model_type $model --engine ort --sam2_dir $sam2_dir --repeats $repeats --onnx_path ${onnx_dir}/${model}_image_decoder_fp32_cpu.onnx --component image_decoder
}

run_gpu()
{
    repeats=$1

    $python convert_to_onnx.py  --sam2_dir $sam2_dir --optimize --use_gpu --dtype fp32
    $python convert_to_onnx.py  --sam2_dir $sam2_dir --optimize --use_gpu --dtype fp16 --demo

    echo "Benchmarking SAM2 model $model image encoder for PyTorch ..."
    $python benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --use_gpu --dtype bf16
    $python benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --use_gpu --dtype fp32

    # Test different torch compile modes on image encoder (none will disable compile and use eager mode).
    for torch_compile_mode in none max-autotune reduce-overhead max-autotune-no-cudagraphs
    do
        $python benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --use_gpu --dtype fp16 --component image_encoder --torch_compile_mode $torch_compile_mode
    done

    echo "Benchmarking SAM2 model $model image decoder for PyTorch ..."
    $python benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --use_gpu --dtype bf16 --component image_decoder
    $python benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --use_gpu --dtype fp32 --component image_decoder
    $python benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --use_gpu --dtype fp16 --component image_decoder

    echo "Benchmarking SAM2 model $model image encoder for ORT ..."
    $python benchmark_sam2.py --model_type $model --engine ort --sam2_dir $sam2_dir --repeats $repeats --onnx_path ${onnx_dir}/${model}_image_encoder_fp16_gpu.onnx --use_gpu --dtype fp16
    $python benchmark_sam2.py --model_type $model --engine ort --sam2_dir $sam2_dir --repeats $repeats --onnx_path ${onnx_dir}/${model}_image_encoder_fp32_gpu.onnx --use_gpu --dtype fp32

    echo "Benchmarking SAM2 model $model image decoder for ORT ..."
    $python benchmark_sam2.py --model_type $model --engine ort --sam2_dir $sam2_dir --repeats $repeats --onnx_path ${onnx_dir}/${model}_image_decoder_fp16_gpu.onnx --component image_decoder --use_gpu --dtype fp16
    $python benchmark_sam2.py --model_type $model --engine ort --sam2_dir $sam2_dir --repeats $repeats --onnx_path ${onnx_dir}/${model}_image_decoder_fp32_gpu.onnx --component image_decoder --use_gpu
}

# Build onnxruntime-gpu from source for profiling.
build_onnxruntime_gpu_for_profiling()
{
    pushd $install_dir
    if ! [ -d onnxruntime ]; then
        git clone https://github.com/microsoft/onnxruntime
    fi
    cd onnxruntime

    # Get the CUDA compute capability of the GPU.
    CUDA_ARCH=$(python3 -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')")

    if [ -n "$CUDA_ARCH" ]; then
        pip install --upgrade pip cmake psutil setuptools wheel packaging ninja numpy==1.26.4
        sh build.sh --config Release --build_dir build/cuda12 --build_shared_lib --parallel \
                --use_cuda --cuda_version 12.5 --cuda_home $install_dir/cuda12.5 \
                --cudnn_home $install_dir/cudnn9.4 \
                --build_wheel --skip_tests \
                --cmake_generator Ninja \
                --compile_no_warning_as_error \
                --enable_cuda_nhwc_ops \
                --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH \
                --cmake_extra_defines onnxruntime_ENABLE_NVTX_PROFILE=ON \
                --enable_cuda_line_info

        pip install build/cuda12/Release/dist/onnxruntime_gpu-*-linux_x86_64.whl numpy==1.26.4
    else
        echo "PyTorch is not installed or No CUDA device found."
        exit 1
    fi

    popd
}

# Run profiling with NVTX.
run_nvtx_profile()
{
    pip install nvtx cuda-python==12.5.0

    # Only trace one device to avoid huge output file size.
    device_id=0

    # Environment variables
    envs="CUDA_VISIBLE_DEVICES=$device_id,ORT_ENABLE_CUDNN_FLASH_ATTENTION=1,LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

    # For cuda graphs, node activities will be collected and CUDA graphs will not be traced as a whole.
    # This may cause significant runtime overhead. But it is useful to understand the performance of individual nodes.
    cuda_graph_trace=node

    for engine in ort torch
    do
        for component in image_encoder image_decoder
        do
            sudo $install_dir/cuda12.5/bin/nsys profile --capture-range=nvtx --nvtx-capture='one_run' \
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
run_torch_profile()
{
    for component in image_encoder image_decoder
    do
        $python benchmark_sam2.py --model_type $model --engine torch \
                                  --sam2_dir  $sam2_dir --warm_up 1 --repeats 0 \
                                  --component $component \
                                  --use_gpu --dtype fp16 --enable_torch_profile
    done
}

if ! [ -v CONDA_PREFIX ]; then
    echo "Please activate conda environment before running this script."
    exit 1
fi

# Check whether nvidia-smi is available to determine whether to install GPU or CPU version.
if [ "$cpu_or_gpu" = "gpu" ]; then
    install_gpu
else
    install_cpu
fi

install_sam2

download_test_image

if ! [ -f sam2_${cpu_or_gpu}.csv ]; then
    if [ "$cpu_or_gpu" = "gpu" ]; then
        echo "Running GPU benchmark..."
        run_gpu 1000
    else
        echo "Running CPU benchmark..."
        run_cpu 100
    fi

    cat benchmark*.csv > combined_csv
    awk '!x[$0]++' combined_csv > sam2_${cpu_or_gpu}.csv
    rm combined_csv

    echo "Benchmarking SAM2 model $model results are saved in sam2_${cpu_or_gpu}.csv"
else
    echo "sam2_${cpu_or_gpu}.csv already exists, skipping benchmarking..."
fi

if [ "$cpu_or_gpu" = "gpu" ]; then
    echo "Running GPU profiling..."
    if ! [ -f sam2_fp16_profile_image_decoder_ort_${cpu_or_gpu}.nsys-rep ]; then
        rm -f *.nsys-rep
        rm -f *.sqlite
        build_onnxruntime_gpu_for_profiling
        run_nvtx_profile
    else
        echo "sam2_fp16_profile_ort.nsys-rep already exists, skipping GPU profiling..."
    fi

    run_torch_profile
fi
