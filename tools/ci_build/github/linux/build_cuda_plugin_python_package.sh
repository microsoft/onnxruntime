#!/bin/bash
set -e -x

DOCKER_IMAGE="onnxruntimecuda128pluginbuildx64"
PYTHON_EXE="/opt/python/cp312-cp312/bin/python3.12"
VERSION=""
PACKAGE_NAME=""

while getopts "i:p:v:n:" parameter_Option
do case "${parameter_Option}"
in
i) DOCKER_IMAGE=${OPTARG};;
p) PYTHON_EXE=${OPTARG};;
v) VERSION=${OPTARG};;
n) PACKAGE_NAME=${OPTARG};;
*) echo "Usage: $0 -i <docker_image> -p <python_exe_path> -v <version> -n <package_name>"
   exit 1;;
esac
done

if [ -z "$VERSION" ]; then
  echo "ERROR: Version is required. Use -v <version>"
  exit 1
fi

if [ -z "$PACKAGE_NAME" ]; then
  echo "ERROR: Package name is required. Use -n <package_name>"
  exit 1
fi

PYTHON_BIN_DIR=$(dirname "${PYTHON_EXE}")

docker run --rm \
    --volume "${BUILD_SOURCESDIRECTORY}:/onnxruntime_src" \
    --volume "${BUILD_BINARIESDIRECTORY}:/build" \
    --volume "${BUILD_ARTIFACTSTAGINGDIRECTORY}:/staging" \
    --env PIP_INDEX_URL \
    --env "ORT_CUDA_PLUGIN_EP_VERSION=${VERSION}" \
    --env "ORT_CUDA_PLUGIN_EP_PACKAGE_NAME=${PACKAGE_NAME}" \
    "$DOCKER_IMAGE" \
    /bin/bash -c '
      set -e -x
      PATH="'"${PYTHON_BIN_DIR}"'":$PATH
      "'"${PYTHON_EXE}"'" -m pip install -r /onnxruntime_src/plugin-ep-cuda/python/requirements-build-wheel.txt
      "'"${PYTHON_EXE}"'" /onnxruntime_src/plugin-ep-cuda/python/build_wheel.py \
        --binary_dir /build/plugin_artifacts/bin \
        --version "$ORT_CUDA_PLUGIN_EP_VERSION" \
        --package_name "$ORT_CUDA_PLUGIN_EP_PACKAGE_NAME" \
        --output_dir /staging/python
    '
