#!/bin/bash

# 1. build ORT
../build.sh --parallel --build_shared_lib --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --use_tensorrt --tensorrt_home /usr/lib/x86_64-linux-gnu/ --config Release --skip_tests --skip_submodule_sync --cmake_extra_defines 'CMAKE_CUDA_ARCHITECTURES=80' --allow_running_as_root --compile_no_warning_as_error

# 2. build test module
cd outTreeEp && rm -r build && mkdir -p build && cd build
cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Release -DORT_HOME=/home/yifanl/onnxruntime
cmake --build ./
cd ../../

cd outTreeEp_kernel && rm -r build && mkdir -p build && cd build
cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Release -DORT_HOME=/home/yifanl/onnxruntime
cmake --build ./
cd ../../

cd tensorRTEp && rm -r build && mkdir -p build && cd build
# Download TensorRT with correct cuda version selected
cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DTENSORRT_HOME=/home/yifanl/TensorRT-10.4.0.26 -DORT_HOME=/home/yifanl/onnxruntime/
cmake --build ./
cd ../../

cp ../build/Linux/Release/libonnxruntime.so c_test
cd c_test && rm -r build && mkdir -p build && cd build
cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Release -DORT_HOME=/home/yifanl/onnxruntime
cmake --build ./
cd ../../

cd c_test && bash sanityTests.sh