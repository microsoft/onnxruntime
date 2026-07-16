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

docker run \
    --rm \
    --volume "${BUILD_BINARIESDIRECTORY}:/build" \
    --volume "${BUILD_SOURCESDIRECTORY}:/onnxruntime_src" \
    --volume "${HOME}/.gitconfig:/home/onnxruntimedev/.gitconfig:ro" \
    --volume "${HOME}/.gradle:/home/onnxruntimedev/.gradle" \
    --volume "${HOME}/.m2:/home/onnxruntimedev/.m2:ro" \
    --volume "${HOME}/.onnx:/home/onnxruntimedev/.onnx" \
    --volume "${NPM_CONFIG_USERCONFIG}:/tmp/.npmrc:ro" \
    --volume /data/models:/build/models:ro \
    --volume /data/onnx:/data/onnx:ro \
    -e BUILD_BUILDNUMBER \
    -e NIGHTLY_BUILD \
    -e NPM_CONFIG_USERCONFIG=/tmp/.npmrc \
    -e PIP_INDEX_URL \
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
        --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF ${EXTRA_CMAKE_DEFINES}"
