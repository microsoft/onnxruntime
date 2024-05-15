#!/bin/bash
set -e -x
docker run --gpus all -e NVIDIA_VISIBLE_DEVICES=all --rm --volume \
$BUILD_SOURCESDIRECTORY:/onnxruntime_src --volume $BUILD_BINARIESDIRECTORY:/build \
--volume /data/models:/build/models:ro --volume /data/onnx:/data/onnx:ro -e NIGHTLY_BUILD onnxruntimecuda${CUDA_VERSION_MAJOR}build \
/usr/bin/python3.9 /onnxruntime_src/tools/ci_build/build.py --enable_lto --build_java --build_nodejs --build_dir /build --config Release \
--skip_submodule_sync  --parallel --use_binskim_compliant_compile_flags --build_shared_lib --use_cuda --cuda_version=$CUDA_VERSION \
--cuda_home=/usr/local/cuda-$CUDA_VERSION --cudnn_home=/usr/local/cuda-$CUDA_VERSION \
--cmake_extra_defines 'CMAKE_CUDA_ARCHITECTURES=60;61;70;75;80'
