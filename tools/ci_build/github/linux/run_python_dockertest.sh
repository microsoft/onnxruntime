#!/bin/bash
set -e -x
BUILD_CONFIG="Release"

while getopts "i:d:x:c:" parameter_Option
do case "${parameter_Option}"
in
i) DOCKER_IMAGE=${OPTARG};;
d) DEVICE=${OPTARG};;
c) BUILD_CONFIG=${OPTARG};;
esac
done

if [ $DEVICE = "GPU" ]; then
  ADDITIONAL_DOCKER_PARAMETER="--gpus all"
fi

mkdir -p $HOME/.onnx
docker run -e SYSTEM_COLLECTIONURI --rm \
    --volume /data/onnx:/data/onnx:ro \
    --volume $BUILD_SOURCESDIRECTORY:/onnxruntime_src \
    --volume $BUILD_BINARIESDIRECTORY:/build \
    --volume /data/models:/build/models:ro \
    --volume $HOME/.onnx:/home/onnxruntimedev/.onnx \
    -e NPM_CONFIG_USERCONFIG=/tmp/.npmrc --volume $AGENT_TEMPDIRECTORY/.npmrc:/tmp/.npmrc:ro \
    --volume $HOME/.m2:/home/onnxruntimedev/.m2:ro \
    --volume $HOME/.gradle:/home/onnxruntimedev/.gradle \
    -w /onnxruntime_src \
    -e NIGHTLY_BUILD \
    -e BUILD_BUILDNUMBER \
    $ADDITIONAL_DOCKER_PARAMETER \
    $DOCKER_IMAGE tools/ci_build/github/linux/run_python_tests.sh -d $DEVICE -c $BUILD_CONFIG
