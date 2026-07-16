#!/bin/bash
set -e -x

if [ "$CUDA_VERSION" == "12.8" ]; then
    CUDA_ARCHS="60-real;70-real;75-real;80-real;86-real;90-real;120-real;120-virtual"
elif [ "$CUDA_VERSION" == "13.0" ]; then
    CUDA_ARCHS="75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"
else
    echo "Error: Unrecognized CUDA_VERSION: $CUDA_VERSION"
    exit 1
fi

mkdir -p "$HOME/.onnx"
docker run \
    --network=host \
    --rm \
    --volume "/data/models:/build/models:ro" \
    --volume "/data/onnx:/data/onnx:ro" \
    --volume "${BUILD_BINARIESDIRECTORY}:/build" \
    --volume "${BUILD_SOURCESDIRECTORY}:/onnxruntime_src" \
    --volume "${HOME}/.gitconfig:/home/onnxruntimedev/.gitconfig:ro" \
    --volume "${HOME}/.gradle:/home/onnxruntimedev/.gradle" \
    --volume "${HOME}/.m2:/home/onnxruntimedev/.m2:ro" \
    --volume "${HOME}/.onnx:/home/onnxruntimedev/.onnx" \
    --volume "${NPM_CONFIG_USERCONFIG}:/tmp/.npmrc:ro" \
    -e NIGHTLY_BUILD \
    -e NPM_CONFIG_USERCONFIG=/tmp/.npmrc \
    -e PIP_INDEX_URL \
    -e SYSTEM_COLLECTIONURI \
    "onnxruntimecuda${CUDA_VERSION_MAJOR}xtrt86build" \
    /bin/bash -c "\
        /usr/bin/python3 /onnxruntime_src/tools/ci_build/build.py \
            --build_dir /build \
            --build_nodejs \
            --build_shared_lib \
            --cmake_extra_defines 'CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}' \
            --cmake_extra_defines 'CMAKE_MESSAGE_LOG_LEVEL=VERBOSE' \
            --config Release \
            --cuda_home '/usr/local/cuda-${CUDA_VERSION}' \
            --cuda_version '${CUDA_VERSION}' \
            --cudnn_home '/usr' \
            --flash_nvcc_threads 1 \
            --nvcc_threads 1 \
            --parallel \
            --skip_submodule_sync \
            --skip_tests \
            --tensorrt_home '/usr' \
            --use_binskim_compliant_compile_flags \
            --use_tensorrt \
            --use_vcpkg \
            --use_vcpkg_ms_internal_asset_cache \
            --use_webgpu \
        && cd /build/Release \
        && make install DESTDIR=/build/installed"
