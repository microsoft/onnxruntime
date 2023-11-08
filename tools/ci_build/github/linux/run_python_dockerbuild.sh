#!/bin/bash
set -e -x
BUILD_CONFIG="Release"

while getopts "i:d:x:c:" parameter_Option
do case "${parameter_Option}"
in
i) DOCKER_IMAGE=${OPTARG};;
d) DEVICE=${OPTARG};;
x) BUILD_EXTR_PAR=${OPTARG};;
c) BUILD_CONFIG=${OPTARG};;
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
    $DOCKER_IMAGE tools/ci_build/github/linux/build_linux_python_package_cuda12.sh -d $DEVICE -c $BUILD_CONFIG -x $BUILD_EXTR_PAR

sudo rm -rf $BUILD_BINARIESDIRECTORY/$BUILD_CONFIG/onnxruntime $BUILD_BINARIESDIRECTORY/$BUILD_CONFIG/pybind11 \
    $BUILD_BINARIESDIRECTORY/$BUILD_CONFIG/models $BUILD_BINARIESDIRECTORY/$BUILD_CONFIG/_deps \
    $BUILD_BINARIESDIRECTORY/$BUILD_CONFIG/CMakeFiles
cd $BUILD_BINARIESDIRECTORY/$BUILD_CONFIG
find -executable -type f > $BUILD_BINARIESDIRECTORY/$BUILD_CONFIG/perms.txt
