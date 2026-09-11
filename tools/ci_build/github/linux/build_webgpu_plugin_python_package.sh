#!/bin/bash
set -e -x

# Build the onnxruntime-ep-webgpu Python wheel inside Docker.
# The Docker container provides a manylinux-compatible environment
# with the correct Python version and auditwheel support.

DOCKER_IMAGE="onnxruntimewebgpuplugin"
VERSION=""

while getopts "i:v:" parameter_Option
do case "${parameter_Option}"
in
i) DOCKER_IMAGE=${OPTARG};;
v) VERSION=${OPTARG};;
*) echo "Usage: $0 -i <docker_image> -v <version>"
   exit 1;;
esac
done

if [ -z "$VERSION" ]; then
  echo "ERROR: Version is required. Use -v <version>"
  exit 1
fi

docker run --rm \
    --volume "${BUILD_SOURCESDIRECTORY}:/onnxruntime_src" \
    --volume "${BUILD_BINARIESDIRECTORY}:/build" \
    --volume "${BUILD_ARTIFACTSTAGINGDIRECTORY}:/staging" \
    --env "PIP_INDEX_URL=${PIP_INDEX_URL}" \
    --env "ORT_WEBGPU_PLUGIN_EP_VERSION=${VERSION}" \
    "$DOCKER_IMAGE" \
    /bin/bash -c '
      set -e -x
      python3 -m ensurepip
      python3 -m pip install -r /onnxruntime_src/plugin-ep-webgpu/python/requirements-build-wheel.txt
      python3 /onnxruntime_src/plugin-ep-webgpu/python/build_wheel.py \
        --binary_dir /build/plugin_artifacts/bin \
        --version "$ORT_WEBGPU_PLUGIN_EP_VERSION" \
        --output_dir /staging/python
    '
