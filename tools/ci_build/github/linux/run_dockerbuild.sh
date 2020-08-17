#!/bin/bash
set -e -o -x
id
SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"
SOURCE_ROOT=$(realpath $SCRIPT_DIR/../../../../)
CUDA_VER=cuda10.1-cudnn7.6
YOCTO_VERSION="4.19"

while getopts c:o:d:r:p:x:a:v:y: parameter_Option
do case "${parameter_Option}"
in
#android, ubuntu16.04, manylinux2010, ubuntu18.04, CentOS7
o) BUILD_OS=${OPTARG};;
#cpu, gpu, tensorrt
d) BUILD_DEVICE=${OPTARG};;
r) BUILD_DIR=${OPTARG};;
#python version: 3.6 3.7 (absence means default 3.6)
p) PYTHON_VER=${OPTARG};;
# "--build_wheel --use_openblas"
x) BUILD_EXTR_PAR=${OPTARG};;
# "cuda10.0-cudnn7.3, cuda9.1-cudnn7.1"
c) CUDA_VER=${OPTARG};;
# x86 or other, only for ubuntu16.04 os
a) BUILD_ARCH=${OPTARG};;
# openvino version tag: 2020.2 (OpenVINO EP 2.0 supports version starting 2020.2)
v) OPENVINO_VERSION=${OPTARG};;
# YOCTO 4.19 + ACL 19.05, YOCTO 4.14 + ACL 19.02
y) YOCTO_VERSION=${OPTARG};;
esac
done

EXIT_CODE=1
PYTHON_VER=${PYTHON_VER:=3.6}
echo "bo=$BUILD_OS bd=$BUILD_DEVICE bdir=$BUILD_DIR pv=$PYTHON_VER bex=$BUILD_EXTR_PAR"

# If in docker group, call "docker". Otherwise, call "sudo docker".
if id -Gnz | grep -zq "^docker$" ; then
    DOCKER_CMD=docker
else
    DOCKER_CMD="sudo --preserve-env docker"
fi

cd $SCRIPT_DIR/docker
if [ $BUILD_OS = "android" ]; then
    IMAGE="android"
    DOCKER_FILE=Dockerfile.ubuntu_for_android
    $DOCKER_CMD build --pull -t "onnxruntime-$IMAGE" --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER} -f $DOCKER_FILE .
elif [ $BUILD_OS = "manylinux2010" ]; then
    if [ $BUILD_DEVICE = "gpu" ]; then
        IMAGE="manylinux2010-cuda10.1"
        DOCKER_FILE=Dockerfile.manylinux2010_gpu
    else
        IMAGE="manylinux2010"
        DOCKER_FILE=Dockerfile.manylinux2010
    fi
    $DOCKER_CMD build --pull -t "onnxruntime-$IMAGE" --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER} -f $DOCKER_FILE .
elif [ $BUILD_OS = "centos7" ]; then
    IMAGE="centos7"
    DOCKER_FILE=Dockerfile.centos
    $DOCKER_CMD build --pull -t "onnxruntime-$IMAGE" --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER} -f $DOCKER_FILE .
elif [ $BUILD_OS = "yocto" ]; then
    IMAGE="arm-yocto-$YOCTO_VERSION"
    DOCKER_FILE=Dockerfile.ubuntu_for_arm
    # ACL 19.05 need yocto 4.19
    TOOL_CHAIN_SCRIPT=fsl-imx-xwayland-glibc-x86_64-fsl-image-qt5-aarch64-toolchain-4.19-warrior.sh
    if [ $YOCTO_VERSION = "4.14" ]; then
        TOOL_CHAIN_SCRIPT=fsl-imx-xwayland-glibc-x86_64-fsl-image-qt5-aarch64-toolchain-4.14-sumo.sh
    fi
    $DOCKER_CMD build -t "onnxruntime-$IMAGE" --build-arg TOOL_CHAIN=$TOOL_CHAIN_SCRIPT --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER} -f $DOCKER_FILE .
else
    if [ $BUILD_DEVICE = "gpu" ]; then
        IMAGE="$BUILD_OS-$CUDA_VER"
        DOCKER_FILE=Dockerfile.ubuntu_gpu
        if [ $CUDA_VER = "cuda9.1-cudnn7.1" ]; then
        DOCKER_FILE=Dockerfile.ubuntu_gpu_cuda9
        fi
        $DOCKER_CMD build --pull -t "onnxruntime-$IMAGE" --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER} --build-arg BUILD_EXTR_PAR="${BUILD_EXTR_PAR}" -f $DOCKER_FILE .
    elif [ $BUILD_DEVICE = "tensorrt" ]; then
        # TensorRT container release 20.07
        IMAGE="$BUILD_OS-cuda11.0-cudnn8.0-tensorrt7.1"
        DOCKER_FILE=Dockerfile.ubuntu_tensorrt
        $DOCKER_CMD build --pull -t "onnxruntime-$IMAGE" --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER} -f $DOCKER_FILE .
    elif [ $BUILD_DEVICE = "openvino" ]; then
        IMAGE="$BUILD_OS-openvino"
        DOCKER_FILE=Dockerfile.ubuntu_openvino
        $DOCKER_CMD build --pull -t "onnxruntime-$IMAGE" --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER} --build-arg OPENVINO_VERSION=${OPENVINO_VERSION} -f $DOCKER_FILE .
    else
        IMAGE="$BUILD_OS"
        if [ $BUILD_ARCH = "x86" ]; then
            IMAGE="$IMAGE.x86"
            $DOCKER_CMD build --pull -t "onnxruntime-$IMAGE" --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER} -f Dockerfile.ubuntu_x86 .
        else
            $DOCKER_CMD build --pull -t "onnxruntime-$IMAGE" --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER} -f Dockerfile.ubuntu .
        fi
    fi
fi

set +e
mkdir -p ~/.cache/onnxruntime
mkdir -p ~/.onnx

if [ -z "$NIGHTLY_BUILD" ]; then
    set NIGHTLY_BUILD=0
fi

if [ $BUILD_DEVICE = "cpu" ] || [ $BUILD_DEVICE = "ngraph" ] || [ $BUILD_DEVICE = "openvino" ] || [ $BUILD_DEVICE = "nnapi" ] || [ $BUILD_DEVICE = "arm" ]; then
    RUNTIME=
else
    RUNTIME="--gpus all"
fi

DOCKER_RUN_PARAMETER="--name onnxruntime-$BUILD_DEVICE \
                      --volume $SOURCE_ROOT:/onnxruntime_src \
                      --volume $BUILD_DIR:/build \
                      --volume /data/models:/build/models:ro \
                      --volume $HOME/.cache/onnxruntime:/home/onnxruntimedev/.cache/onnxruntime \
                      --volume $HOME/.onnx:/home/onnxruntimedev/.onnx"
if [ $BUILD_DEVICE = "openvino" ] && [[ $BUILD_EXTR_PAR == *"--use_openvino GPU_FP"* ]]; then
    DOCKER_RUN_PARAMETER="$DOCKER_RUN_PARAMETER --device /dev/dri:/dev/dri"
fi

if [[ $BUILD_EXTR_PAR = *--enable_training_python_frontend_e2e_tests* ]]; then
    DOCKER_RUN_PARAMETER="$DOCKER_RUN_PARAMETER --volume /bert_data/hf_data:/bert_data/hf_data"
    # DOCKER_RUN_PARAMETER="$DOCKER_RUN_PARAMETER -u0"
fi

$DOCKER_CMD rm -f "onnxruntime-$BUILD_DEVICE" || true
$DOCKER_CMD run $RUNTIME -h $HOSTNAME $DOCKER_RUN_PARAMETER \
    -e NIGHTLY_BUILD \
    "onnxruntime-$IMAGE" \
    /bin/bash /onnxruntime_src/tools/ci_build/github/linux/run_build.sh \
    -d $BUILD_DEVICE -x "$BUILD_EXTR_PAR" -o $BUILD_OS -y $YOCTO_VERSION &
wait $!

EXIT_CODE=$?

set -e
exit $EXIT_CODE
