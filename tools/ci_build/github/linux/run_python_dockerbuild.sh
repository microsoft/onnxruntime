#!/bin/bash
set -e -x
while getopts "x:i:" parameter_Option
do case "${parameter_Option}"
in
i) DOCKER_IMAGE=${OPTARG};;
x) BUILD_EXTR_PAR=${OPTARG};;
esac
done

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
    $DOCKER_IMAGE tools/ci_build/github/linux/build_linux_arm64_python_package.sh $BUILD_EXTR_PAR

sudo rm -rf $BUILD_BINARIESDIRECTORY/Release/onnxruntime $BUILD_BINARIESDIRECTORY/Release/pybind11 \
    $BUILD_BINARIESDIRECTORY/Release/models $BUILD_BINARIESDIRECTORY/Release/_deps \
    $BUILD_BINARIESDIRECTORY/Release/CMakeFiles
cd $BUILD_BINARIESDIRECTORY/Release
find -executable -type f > $BUILD_BINARIESDIRECTORY/Release/perms.txt
