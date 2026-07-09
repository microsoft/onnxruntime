#!/bin/bash
set -euxo pipefail

if [ "$CUDA_VERSION" == "12.8" ]; then
    CUDA_ARCHS="60-real;70-real;75-real;80-real;90a-real;90-virtual"
elif [ "$CUDA_VERSION" == "13.0" ]; then
    CUDA_ARCHS="75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"
else
    echo "Error: Unrecognized CUDA_VERSION: $CUDA_VERSION"
    exit 1
fi

# HACK: so the docker user can write to the caches
mkdir -p "${CCACHE_DIR}"
mkdir -p "${VCPKG_DEFAULT_BINARY_CACHE}"
chmod -R 777 "${CCACHE_DIR}"
chmod -R 777 "${VCPKG_DEFAULT_BINARY_CACHE}"

docker run --rm \
    --volume "${BUILD_BINARIESDIRECTORY}:/build" \
    --volume "${BUILD_SOURCESDIRECTORY}:/onnxruntime_src" \
    --volume "${CCACHE_DIR}:/cache/ccache" \
    --volume "${HOME}/.gradle:/home/onnxruntimedev/.gradle" \
    --volume "${HOME}/.m2:/home/onnxruntimedev/.m2:ro" \
    --volume "${NPM_CONFIG_USERCONFIG}:/tmp/.npmrc:ro" \
    --volume "${VCPKG_DEFAULT_BINARY_CACHE}:/cache/vcpkg" \
    -e CCACHE_DIR=/cache/ccache \
    -e NIGHTLY_BUILD "onnxruntimecuda${CUDA_VERSION_MAJOR}build" \
    -e NPM_CONFIG_USERCONFIG=/tmp/.npmrc \
    -e PIP_INDEX_URL \
    -e SYSTEM_COLLECTIONURI \
    -e VCPKG_DEFAULT_BINARY_CACHE=/cache/vcpkg \
    "onnxruntimecuda${CUDA_VERSION_MAJOR}build" \
    /bin/bash -c " \
ccache --zero-stats \
&& /usr/bin/python3 /onnxruntime_src/tools/ci_build/build.py \
    --build_dir /build \
    --build_java \
    --build_nodejs \
    --build_shared_lib \
    --cmake_extra_defines 'CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}' 'onnxruntime_USE_FPA_INTB_GEMM=OFF' \
    --config Release \
    --cuda_home     '/usr/local/cuda-$CUDA_VERSION' \
    --cuda_version  '$CUDA_VERSION' \
    --cudnn_home    '/usr/local/cuda-$CUDA_VERSION' \
    --enable_lto \
    --flash_nvcc_threads 1 \
    --nvcc_threads 1 \
    --parallel \
    --skip_submodule_sync \
    --skip_tests \
    --use_binskim_compliant_compile_flags \
    --use_cache \
    --use_cuda \
    --use_vcpkg \
    --use_vcpkg_ms_internal_asset_cache \
&& ccache --show-stats -vv \
&& cd /build/Release \
&& make install DESTDIR=/build/installed"
