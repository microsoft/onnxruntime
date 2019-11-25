#!/bin/bash
set -e -o -x

SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"
SOURCE_ROOT=$(realpath $SCRIPT_DIR/../../../../)
CUDA_VER=cuda10.0-cudnn7.3

while getopts c:o:d:r:p:x:a: parameter_Option
do case "${parameter_Option}"
in
#android, ubuntu16.04
o) BUILD_OS=${OPTARG};;
#cpu, gpu, tensorrt
d) BUILD_DEVICE=${OPTARG};;
r) BUILD_DIR=${OPTARG};;
#python version: 3.6 3.7 (absence means default 3.5)
p) PYTHON_VER=${OPTARG};;
# "--build_wheel --use_openblas"
x) BUILD_EXTR_PAR=${OPTARG};;
# "cuda10.0-cudnn7.3, cuda9.1-cudnn7.1"
c) CUDA_VER=${OPTARG};;
# x86 or other, only for ubuntu16.04 os
a) BUILD_ARCH=${OPTARG};;
esac
done

EXIT_CODE=1
PYTHON_VER=${PYTHON_VER:=3.5}
echo "bo=$BUILD_OS bd=$BUILD_DEVICE bdir=$BUILD_DIR pv=$PYTHON_VER bex=$BUILD_EXTR_PAR"

IMAGE=ubuntu16.04

cd $SCRIPT_DIR/docker
docker build --pull -t "onnxruntime-server-$IMAGE" --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg OS_VERSION=16.04 --build-arg PYTHON_VERSION=${PYTHON_VER} -f Dockerfile.ubuntu_server .


set +e
mkdir -p ~/.cache/onnxruntime
mkdir -p ~/.onnx
mkdir -p ~/.cache/go

if [ -z "$NIGHTLY_BUILD" ]; then
set NIGHTLY_BUILD=0
fi

if [ $BUILD_DEVICE = "cpu" ] || [ $BUILD_DEVICE = "ngraph" ]; then
    docker rm -f "onnxruntime-$BUILD_DEVICE" || true
    docker run -h $HOSTNAME \
        --name "onnxruntime-$BUILD_DEVICE" \
        --volume "$SOURCE_ROOT:/onnxruntime_src" \
        --volume "$BUILD_DIR:/build" \
        --volume "$HOME/.cache/onnxruntime:/home/onnxruntimedev/.cache/onnxruntime" \
        --volume "$HOME/.onnx:/home/onnxruntimedev/.onnx" \
        --volume "$HOME/.cache/go:/home/onnxruntimedev/.cache/go" \
        -e NIGHTLY_BUILD \
        -e GOCACHE=/home/onnxruntimedev/.cache/go \
        "onnxruntime-server-$IMAGE" \
        /bin/bash /onnxruntime_src/tools/ci_build/github/linux/server_run_build.sh \
         -d $BUILD_DEVICE -x "$BUILD_EXTR_PAR" -o $BUILD_OS &
else
    docker rm -f "onnxruntime-$BUILD_DEVICE" || true
    nvidia-docker run --rm -h $HOSTNAME \
        --rm \
        --name "onnxruntime-$BUILD_DEVICE" \
        --volume "$SOURCE_ROOT:/onnxruntime_src" \
        --volume "$BUILD_DIR:/build" \
        --volume "$HOME/.cache/onnxruntime:/home/onnxruntimedev/.cache/onnxruntime" \
        --volume "$HOME/.onnx:/home/onnxruntimedev/.onnx" \
        --volume "$HOME/.cache/go:/home/onnxruntimedev/.cache/go" \
        -e NIGHTLY_BUILD \
        -e GOCACHE=/home/onnxruntimedev/.cache/go \
        "onnxruntime-server-$IMAGE" \
        /bin/bash /onnxruntime_src/tools/ci_build/github/linux/server_run_build.sh \
        -d $BUILD_DEVICE -x "$BUILD_EXTR_PAR" -o $BUILD_OS &
fi
wait $!

EXIT_CODE=$?

set -e
exit $EXIT_CODE
