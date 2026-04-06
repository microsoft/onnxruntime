#!/bin/bash
set -e -x

# Build WebGPU plugin shared library for Linux inside Docker.
# This script follows the same pattern as build_nodejs_package.sh.

BUILD_CONFIG="Release"
DOCKER_IMAGE="onnxruntimewebgpuplugin"

while getopts "i:c:" parameter_Option
do case "${parameter_Option}"
in
i) DOCKER_IMAGE=${OPTARG};;
c) BUILD_CONFIG=${OPTARG};;
*) echo "Usage: $0 -i <docker_image> [-c <build_config>]"
   exit 1;;
esac
done

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
    /bin/bash -c "/usr/bin/python3 /onnxruntime_src/tools/ci_build/build.py \
        --build_dir /build \
        --config ${BUILD_CONFIG} \
        --skip_submodule_sync \
        --parallel \
        --use_binskim_compliant_compile_flags \
        --use_webgpu shared_lib \
        --wgsl_template static \
        --disable_rtti \
        --enable_lto \
        --enable_onnx_tests \
        --use_vcpkg \
        --use_vcpkg_ms_internal_asset_cache \
        --update \
        --build \
        --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=ON ${EXTRA_CMAKE_DEFINES}"
