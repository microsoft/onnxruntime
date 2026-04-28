#!/bin/bash
set -e -x

# Build CUDA plugin shared library for Linux inside Docker.
# This script follows the same pattern as build_webgpu_plugin_package.sh.

BUILD_CONFIG="Release"
DOCKER_IMAGE="onnxruntimecuda128pluginbuild"
PYTHON_EXE="/opt/python/cp312-cp312/bin/python3.12"
CUDA_VERSION="12.8"
CMAKE_CUDA_ARCHS="86"

while getopts "i:c:p:v:a:" parameter_Option
do case "${parameter_Option}"
in
i) DOCKER_IMAGE=${OPTARG};;
c) BUILD_CONFIG=${OPTARG};;
p) PYTHON_EXE=${OPTARG};;
v) CUDA_VERSION=${OPTARG};;
a) CMAKE_CUDA_ARCHS=${OPTARG};;
*) echo "Usage: $0 -i <docker_image> [-c <build_config>] [-p <python_exe_path>] [-v <cuda_version>] [-a <cuda_archs>]"
   exit 1;;
esac
done

PYTHON_BIN_DIR=$(dirname "${PYTHON_EXE}")

# Derive SHORT_CUDA_VERSION (e.g., 12.8 from 12.8, 13.0 from 13.0)
SHORT_CUDA_VERSION="${CUDA_VERSION}"

# Determine CUDA home — prefer versioned path, fall back to /usr/local/cuda
CUDA_HOME="/usr/local/cuda-${SHORT_CUDA_VERSION}"

mkdir -p "${HOME}/.onnx"

docker run --rm \
    --volume /data/onnx:/data/onnx:ro \
    --volume "${BUILD_SOURCESDIRECTORY}:/onnxruntime_src" \
    --volume "${BUILD_BINARIESDIRECTORY}:/build" \
    --volume /data/models:/build/models:ro \
    --volume "${HOME}/.onnx:/home/onnxruntimedev/.onnx" \
    -e NIGHTLY_BUILD \
    -e BUILD_BUILDNUMBER \
    -e SYSTEM_COLLECTIONURI \
    "$DOCKER_IMAGE" \
    /bin/bash -c "
        CUDA_HOME=${CUDA_HOME}
        if [ ! -d \"\${CUDA_HOME}\" ] && [ -d /usr/local/cuda ]; then
            CUDA_HOME=/usr/local/cuda
        fi
        PATH=${PYTHON_BIN_DIR}:\$PATH \
        ${PYTHON_EXE} /onnxruntime_src/tools/ci_build/build.py \
        --build_dir /build \
        --config ${BUILD_CONFIG} \
        --skip_submodule_sync \
        --parallel \
        --nvcc_threads 1 \
        --use_binskim_compliant_compile_flags \
        --use_cuda \
        --cuda_version=${SHORT_CUDA_VERSION} \
        --cuda_home=\${CUDA_HOME} \
        --cudnn_home=\${CUDA_HOME} \
        --update \
        --build \
        --use_vcpkg \
        --use_vcpkg_ms_internal_asset_cache \
        --skip_tests \
        --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHS}' \
        --cmake_extra_defines onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON \
        ${EXTRA_CMAKE_DEFINES:+--cmake_extra_defines ${EXTRA_CMAKE_DEFINES}}"
