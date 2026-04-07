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
    /bin/bash /onnxruntime_src/tools/ci_build/github/linux/build_webgpu_plugin_package_inner.sh \
        "${BUILD_CONFIG}" "${EXTRA_CMAKE_DEFINES}"
