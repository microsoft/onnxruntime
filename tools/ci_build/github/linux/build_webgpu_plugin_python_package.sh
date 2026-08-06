#!/bin/bash
set -e -x

# Build the onnxruntime-ep-webgpu Python wheel inside Docker.
# The Docker container provides a manylinux-compatible environment
# with the correct Python version and auditwheel support.

DOCKER_IMAGE="onnxruntimewebgpuplugin"
VERSION=""
BUILD_COMMIT=""

while getopts "i:v:c:" parameter_Option
do case "${parameter_Option}"
in
i) DOCKER_IMAGE=${OPTARG};;
v) VERSION=${OPTARG};;
c) BUILD_COMMIT=${OPTARG};;
*) echo "Usage: $0 -i <docker_image> -v <version> -c <build_commit>"
   exit 1;;
esac
done

if [ -z "$VERSION" ]; then
  echo "ERROR: Version is required. Use -v <version>"
  exit 1
fi

if [ -z "$BUILD_COMMIT" ]; then
  echo "ERROR: Build commit is required. Use -c <build_commit>"
  exit 1
fi

docker run --rm \
    --volume "${BUILD_SOURCESDIRECTORY}:/onnxruntime_src" \
    --volume "${BUILD_BINARIESDIRECTORY}:/build" \
    --volume "${BUILD_ARTIFACTSTAGINGDIRECTORY}:/staging" \
    --env "PIP_INDEX_URL=${PIP_INDEX_URL}" \
    --env "ORT_WEBGPU_PLUGIN_EP_VERSION=${VERSION}" \
    --env "ORT_WEBGPU_PLUGIN_EP_BUILD_COMMIT=${BUILD_COMMIT}" \
    "$DOCKER_IMAGE" \
    /bin/bash -c '
      set -e -x
      python3 -m ensurepip
      python3 -m pip install -r /onnxruntime_src/plugin-ep-webgpu/python/requirements-build-wheel.txt
      python3 /onnxruntime_src/plugin-ep-webgpu/python/build_wheel.py \
        --binary_dir /build/plugin_artifacts/bin \
        --version "$ORT_WEBGPU_PLUGIN_EP_VERSION" \
        --build_commit "$ORT_WEBGPU_PLUGIN_EP_BUILD_COMMIT" \
        --output_dir /staging/python
    '
