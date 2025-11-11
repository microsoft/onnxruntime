#!/bin/bash
set -e -x

if [ "$CUDA_VERSION" == "12.8" ]; then
    CUDA_ARCHS="60-real;70-real;75-real;80-real;90a-real;90-virtual"
elif [ "$CUDA_VERSION" == "13.0" ]; then
    CUDA_ARCHS="75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"
else
    echo "Error: Unrecognized CUDA_VERSION: $CUDA_VERSION"
    exit 1
fi

docker run -e SYSTEM_COLLECTIONURI --rm --volume \
$BUILD_SOURCESDIRECTORY:/onnxruntime_src --volume $BUILD_BINARIESDIRECTORY:/build \
-e NIGHTLY_BUILD onnxruntimecuda${CUDA_VERSION_MAJOR}build \
/bin/bash -c "/usr/bin/python3 /onnxruntime_src/tools/ci_build/build.py --enable_lto --build_java --build_nodejs \
--build_dir /build --config Release --skip_submodule_sync  --parallel --use_binskim_compliant_compile_flags --build_shared_lib \
--use_cuda --cuda_version=$CUDA_VERSION --cuda_home=/usr/local/cuda-$CUDA_VERSION --cudnn_home=/usr/local/cuda-$CUDA_VERSION \
--skip_tests --use_vcpkg --use_vcpkg_ms_internal_asset_cache \
--cmake_extra_defines 'CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}' 'onnxruntime_USE_FPA_INTB_GEMM=OFF' \
&& cd /build/Release && make install DESTDIR=/build/installed"
