#!/bin/bash
set -e -x
docker run --rm --volume \
$BUILD_SOURCESDIRECTORY:/onnxruntime_src --volume $BUILD_BINARIESDIRECTORY:/build -e NIGHTLY_BUILD onnxruntimecuda${CUDA_VERSION_MAJOR}build \
/bin/bash -c "/usr/bin/python3.12 /onnxruntime_src/tools/ci_build/build.py --enable_lto --build_java --build_nodejs --build_dir /build --config Release --skip_submodule_sync  --parallel --use_binskim_compliant_compile_flags --build_shared_lib --use_cuda --cuda_version=$CUDA_VERSION --cuda_home=/usr/local/cuda-$CUDA_VERSION --cudnn_home=/usr/local/cuda-$CUDA_VERSION --skip_tests --cmake_extra_defines 'CMAKE_CUDA_ARCHITECTURES=75;80;90' && cd /build/Release && make install DESTDIR=/build/installed"
