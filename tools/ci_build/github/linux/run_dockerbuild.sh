#!/bin/bash
set -e -o -x
id
SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"
SOURCE_ROOT=$(realpath $SCRIPT_DIR/../../../../)
YOCTO_VERSION="4.19"
INSTALL_DEPS_DISTRIBUTED_SETUP=false
ORTMODULE_BUILD=false
USE_CONDA=false
ALLOW_RELEASED_ONNX_OPSET_ONLY_ENV="ALLOW_RELEASED_ONNX_OPSET_ONLY="$ALLOW_RELEASED_ONNX_OPSET_ONLY
echo "ALLOW_RELEASED_ONNX_OPSET_ONLY environment variable is set as "$ALLOW_RELEASED_ONNX_OPSET_ONLY_ENV

while getopts c:o:d:r:p:x:a:v:y:t:i:mue parameter_Option
do case "${parameter_Option}"
in
#android, ubuntu16.04, ubuntu18.04, CentOS7
o) BUILD_OS=${OPTARG};;
#cpu, gpu, tensorrt
d) BUILD_DEVICE=${OPTARG};;
r) BUILD_DIR=${OPTARG};;
#python version: 3.6 3.7 (absence means default 3.6)
p) PYTHON_VER=${OPTARG};;
# "--build_wheel --use_openblas"
x) BUILD_EXTR_PAR=${OPTARG};;
# x86 or other, only for ubuntu16.04 os
a) BUILD_ARCH=${OPTARG};;
# openvino version tag: 2020.3 (OpenVINO EP 2.0 supports version starting 2020.3)
v) OPENVINO_VERSION=${OPTARG};;
# YOCTO 4.19 + ACL 19.05, YOCTO 4.14 + ACL 19.02
y) YOCTO_VERSION=${OPTARG};;
# an additional name for the resulting docker image (created with "docker tag")
# this is useful for referencing the image outside of this script
t) EXTRA_IMAGE_TAG=${OPTARG};;
# the docker image cache container registry
i) IMAGE_CACHE_CONTAINER_REGISTRY_NAME=${OPTARG};;
# install distributed setup dependencies
m) INSTALL_DEPS_DISTRIBUTED_SETUP=true;;
# install ortmodule specific dependencies
u) ORTMODULE_BUILD=true;;
# install and use conda
e) USE_CONDA=true;;
esac
done

EXIT_CODE=1
PYTHON_VER=${PYTHON_VER:=3.6}
echo "bo=$BUILD_OS bd=$BUILD_DEVICE bdir=$BUILD_DIR pv=$PYTHON_VER bex=$BUILD_EXTR_PAR"

GET_DOCKER_IMAGE_CMD="${SOURCE_ROOT}/tools/ci_build/get_docker_image.py"
if [[ -n "${IMAGE_CACHE_CONTAINER_REGISTRY_NAME}" ]]; then
    GET_DOCKER_IMAGE_CMD="${GET_DOCKER_IMAGE_CMD} --container-registry ${IMAGE_CACHE_CONTAINER_REGISTRY_NAME}"
fi
DOCKER_CMD="docker"

cd $SCRIPT_DIR/docker
if [ $BUILD_OS = "android" ]; then
    IMAGE="android"
    DOCKER_FILE=Dockerfile.ubuntu_for_android
    $GET_DOCKER_IMAGE_CMD --repository "onnxruntime-$IMAGE" \
        --docker-build-args="--build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER}" \
        --dockerfile $DOCKER_FILE --context .
elif [ $BUILD_OS = "centos7" ]; then
    IMAGE="centos7"
    DOCKER_FILE=Dockerfile.centos
    $GET_DOCKER_IMAGE_CMD --repository "onnxruntime-$IMAGE" \
        --docker-build-args="--build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER}" \
        --dockerfile $DOCKER_FILE --context .
elif [ $BUILD_OS = "yocto" ]; then
    IMAGE="arm-yocto-$YOCTO_VERSION"
    DOCKER_FILE=Dockerfile.ubuntu_for_arm
    # ACL 19.05 need yocto 4.19
    TOOL_CHAIN_SCRIPT=fsl-imx-xwayland-glibc-x86_64-fsl-image-qt5-aarch64-toolchain-4.19-warrior.sh
    if [ $YOCTO_VERSION = "4.14" ]; then
        TOOL_CHAIN_SCRIPT=fsl-imx-xwayland-glibc-x86_64-fsl-image-qt5-aarch64-toolchain-4.14-sumo.sh
    fi
    $GET_DOCKER_IMAGE_CMD --repository "onnxruntime-$IMAGE" \
        --docker-build-args="--build-arg TOOL_CHAIN=$TOOL_CHAIN_SCRIPT --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER}" \
        --dockerfile $DOCKER_FILE --context .
else
    if [ $BUILD_DEVICE = "gpu" ]; then
        #This code path is only for training. Inferecing pipeline uses CentOS
        IMAGE="ubuntu_gpu_training"
        INSTALL_DEPS_EXTRA_ARGS="${INSTALL_DEPS_EXTRA_ARGS} -t"
        if [[ $INSTALL_DEPS_DISTRIBUTED_SETUP = true ]]; then
            INSTALL_DEPS_EXTRA_ARGS="${INSTALL_DEPS_EXTRA_ARGS} -m"
        fi
        if [[ $ORTMODULE_BUILD = true ]]; then
            INSTALL_DEPS_EXTRA_ARGS="${INSTALL_DEPS_EXTRA_ARGS} -u"
        fi
        $GET_DOCKER_IMAGE_CMD --repository "onnxruntime-$IMAGE" \
            --docker-build-args="--build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER} --build-arg INSTALL_DEPS_EXTRA_ARGS=\"${INSTALL_DEPS_EXTRA_ARGS}\" --build-arg USE_CONDA=${USE_CONDA}" \
            --dockerfile Dockerfile.ubuntu_gpu_training --context .
    elif [ $BUILD_DEVICE = "tensorrt"* ]; then
        if [ $BUILD_DEVICE = "tensorrt-v7.1" ]; then
            # TensorRT container release 20.07
            IMAGE="$BUILD_OS-cuda11.0-cudnn8.0-tensorrt7.1"
            DOCKER_FILE=Dockerfile.ubuntu_tensorrt7_1
        else
            # TensorRT container release 20.12
            IMAGE="$BUILD_OS-cuda11.1-cudnn8.0-tensorrt7.2"
            DOCKER_FILE=Dockerfile.ubuntu_tensorrt
        fi
        $GET_DOCKER_IMAGE_CMD --repository "onnxruntime-$IMAGE" \
            --docker-build-args="--build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER}" \
            --dockerfile $DOCKER_FILE --context .
    elif [ $BUILD_DEVICE = "openvino" ]; then
        IMAGE="$BUILD_OS-openvino"
        DOCKER_FILE=Dockerfile.ubuntu_openvino
        $GET_DOCKER_IMAGE_CMD --repository "onnxruntime-$IMAGE" \
            --docker-build-args="--build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER} --build-arg OPENVINO_VERSION=${OPENVINO_VERSION}" \
            --dockerfile $DOCKER_FILE --context .
    else
        IMAGE="$BUILD_OS"
        $GET_DOCKER_IMAGE_CMD --repository "onnxruntime-$IMAGE" \
                --docker-build-args="--build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER}" \
                --dockerfile Dockerfile.ubuntu --context .
    fi
fi

if [ -v EXTRA_IMAGE_TAG ]; then
    ${DOCKER_CMD} tag "onnxruntime-$IMAGE" "${EXTRA_IMAGE_TAG}"
fi

set +e
mkdir -p ~/.cache/onnxruntime
mkdir -p ~/.onnx

if [ -z "$NIGHTLY_BUILD" ]; then
    set NIGHTLY_BUILD=0
fi

if [ $BUILD_DEVICE = "cpu" ] || [ $BUILD_DEVICE = "openvino" ] || [ $BUILD_DEVICE = "nnapi" ] || [ $BUILD_DEVICE = "arm" ]; then
    RUNTIME=
else
    RUNTIME="--gpus all"
fi

DOCKER_RUN_PARAMETER="--name onnxruntime-$BUILD_DEVICE \
                      --volume $SOURCE_ROOT:/onnxruntime_src \
                      --volume $BUILD_DIR:/build \
                      --volume /data/models:/build/models:ro \
                      --volume /data/onnx:/data/onnx:ro \
                      --volume $HOME/.cache/onnxruntime:/home/onnxruntimedev/.cache/onnxruntime \
                      --volume $HOME/.onnx:/home/onnxruntimedev/.onnx"
if [ $BUILD_DEVICE = "openvino" ] && [[ $BUILD_EXTR_PAR == *"--use_openvino GPU_FP"* ]]; then
    DOCKER_RUN_PARAMETER="$DOCKER_RUN_PARAMETER --device /dev/dri:/dev/dri"
fi

$DOCKER_CMD rm -f "onnxruntime-$BUILD_DEVICE" || true
$DOCKER_CMD run $RUNTIME -h $HOSTNAME $DOCKER_RUN_PARAMETER \
    -e NIGHTLY_BUILD \
    -e $ALLOW_RELEASED_ONNX_OPSET_ONLY_ENV \
    "onnxruntime-$IMAGE" \
    /bin/bash /onnxruntime_src/tools/ci_build/github/linux/run_build.sh \
    -d $BUILD_DEVICE -x "$BUILD_EXTR_PAR" -o $BUILD_OS -y $YOCTO_VERSION &
wait $!

EXIT_CODE=$?

set -e
exit $EXIT_CODE
