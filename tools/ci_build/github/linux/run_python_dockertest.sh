#!/bin/bash
set -e -x
BUILD_CONFIG="Release"

while getopts "i:d:c:u:" parameter_Option
do case "${parameter_Option}"
in
i) DOCKER_IMAGE=${OPTARG};;
d) DEVICE=${OPTARG};;
c) BUILD_CONFIG=${OPTARG};;
u) CUDA_VERSION=${OPTARG:"11.8"};;
*) echo "Usage: $0 -i <docker_image> -d <GPU|CPU> [-c <build_config>] [-u <cuda_version>]"
   exit 1;;
esac
done

if [ $DEVICE = "GPU" ]; then
  ADDITIONAL_DOCKER_PARAMETER="--gpus all"
fi

mkdir -p $HOME/.onnx
docker run --rm \
    --volume /data/onnx:/data/onnx:ro \
    --volume $BUILD_SOURCESDIRECTORY:/onnxruntime_src \
    --volume $BUILD_BINARIESDIRECTORY:/build \
    --volume /data/models:/build/models:ro \
    --volume $HOME/.onnx:/home/onnxruntimedev/.onnx \
    -w /onnxruntime_src \
    -e NIGHTLY_BUILD \
    -e BUILD_BUILDNUMBER \
    $ADDITIONAL_DOCKER_PARAMETER \
    $DOCKER_IMAGE tools/ci_build/github/linux/run_python_tests.sh -d $DEVICE -c $BUILD_CONFIG -u $CUDA_VERSION