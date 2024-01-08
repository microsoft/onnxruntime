#!/bin/bash
set -e -x
export CFLAGS="-NDEBUG -Wp,-D_FORTIFY_SOURCE=2 -Wp,-D_GLIBCXX_ASSERTIONS -fstack-protector-strong -fstack-clash-protection -fcf-protection -O3 -Wl,--strip-all"
export CXXFLAGS="-NDEBUG -Wp,-D_FORTIFY_SOURCE=2 -Wp,-D_GLIBCXX_ASSERTIONS -fstack-protector-strong -fstack-clash-protection -fcf-protection -O3 -Wl,--strip-all"
mkdir -p $HOME/.onnx
docker run --gpus all -e CFLAGS -e CXXFLAGS -e NVIDIA_VISIBLE_DEVICES=all --rm --volume /data/onnx:/data/onnx:ro --volume $BUILD_SOURCESDIRECTORY:/onnxruntime_src --volume $BUILD_BINARIESDIRECTORY:/build \
--volume /data/models:/build/models:ro --volume $HOME/.onnx:/home/onnxruntimedev/.onnx -e NIGHTLY_BUILD onnxruntimecuda${CUDA_VERSION_MAJOR}xtrt86build \
/opt/python/cp38-cp38/bin/python3 /onnxruntime_src/tools/ci_build/build.py --build_dir /build --config Release \
--skip_submodule_sync --parallel --build_shared_lib --build_java --build_nodejs --use_tensorrt --cuda_version=$CUDA_VERSION --cuda_home=/usr/local/cuda-$CUDA_VERSION --cudnn_home=/usr --tensorrt_home=/usr --cmake_extra_defines 'CMAKE_CUDA_ARCHITECTURES=60;61;70;75;80'
