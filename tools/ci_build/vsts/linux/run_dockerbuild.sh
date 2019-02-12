#!/bin/bash
set -e -o -x

SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"
SOURCE_ROOT=$(realpath $SCRIPT_DIR/../../../../)
CUDA_VER=cuda9.1-cudnn7.1

while getopts c:o:d:r:p:x: parameter_Option
do case "${parameter_Option}"
in
#ubuntu16.04
o) BUILD_OS=${OPTARG};;
#cpu, fpga, gpu
d) BUILD_DEVICE=${OPTARG};;
r) BUILD_DIR=${OPTARG};;
#python version: 3.6 3.7 (absence means default 3.5)
p) PYTHON_VER=${OPTARG};;
# "--build_wheel --use_openblas"
x) BUILD_EXTR_PAR=${OPTARG};;
# "cuda10.0-cudnn7.3, cuda9.1-cudnn7.1"
c) CUDA_VER=${OPTARG};;
esac
done

EXIT_CODE=1

echo "bo=$BUILD_OS bd=$BUILD_DEVICE bdir=$BUILD_DIR pv=$PYTHON_VER bex=$BUILD_EXTR_PAR"

cd $SCRIPT_DIR/docker
if [ $BUILD_DEVICE = "gpu" ]; then
    IMAGE="ubuntu16.04-$CUDA_VER"
    DOCKER_FILE=Dockerfile.ubuntu_gpu
    if [ $CUDA_VER = "cuda9.1-cudnn7.1" ]; then
    DOCKER_FILE=Dockerfile.ubuntu_gpu_cuda9
    fi
    docker build -t "onnxruntime-$IMAGE" --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER} -f $DOCKER_FILE .
elif [ $BUILD_DEVICE = "fpga" ]; then
    IMAGE="ubuntu16.04-fpga"
    docker build -t "onnxruntime-$IMAGE" --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER} -f Dockerfile.ubuntu_fpga .
else
    IMAGE="ubuntu16.04"
    docker build -t "onnxruntime-$IMAGE" --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg OS_VERSION=16.04 --build-arg PYTHON_VERSION=${PYTHON_VER} -f Dockerfile.ubuntu .
fi

set +e

if [ $BUILD_DEVICE = "cpu" ]; then
    docker rm -f "onnxruntime-$BUILD_DEVICE" || true
    docker run -h $HOSTNAME \
        --rm \
        --name "onnxruntime-$BUILD_DEVICE" \
        --volume "$SOURCE_ROOT:/onnxruntime_src" \
        --volume "$BUILD_DIR:/home/onnxruntimedev" \
        --volume "$HOME/.cache/onnxruntime:/home/onnxruntimedev/.cache/onnxruntime" \
        "onnxruntime-$IMAGE" \
        /bin/bash /onnxruntime_src/tools/ci_build/vsts/linux/run_build.sh \
         -d $BUILD_DEVICE -x "$BUILD_EXTR_PAR" &
elif [ $BUILD_DEVICE = "fpga" ]; then
    docker run -h $HOSTNAME \
        --rm --volume "$HOME/.cache/onnxruntime:/home/onnxruntimedev/.cache/onnxruntime" \
        --volume "$SOURCE_ROOT:/onnxruntime_src" \
        --volume "$BUILD_DIR:/home/onnxruntimedev" \
        --device "/dev/catapult0:/dev/catapult0" \
        --device "/dev/catapult1:/dev/catapult1" \
        -e AIINFRA_PAT -e MSAZURE_PAT -e MSCATAPULT_PAT \
        "onnxruntime-$IMAGE" \
        /bin/bash /onnxruntime_src/tools/ci_build/vsts/linux/run_build.sh \
         -d $BUILD_DEVICE -x "$BUILD_EXTR_PAR" &
else
    nvidia-docker run -h $HOSTNAME \
        --rm --volume "$HOME/.cache/onnxruntime:/home/onnxruntimedev/.cache/onnxruntime" \
        --volume "$SOURCE_ROOT:/onnxruntime_src" \
        --volume "$BUILD_DIR:/home/onnxruntimedev" \
        "onnxruntime-$IMAGE" \
        /bin/bash /onnxruntime_src/tools/ci_build/vsts/linux/run_build.sh \
        -d $BUILD_DEVICE -x "$BUILD_EXTR_PAR" &
fi
wait -n

EXIT_CODE=$?

set -e
exit $EXIT_CODE
