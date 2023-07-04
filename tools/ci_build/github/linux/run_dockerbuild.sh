#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used by Linux Multi GPU TensorRT CI Pipeline,Linux OpenVINO CI Pipeline,orttraining-linux-gpu-ci-pipeline
#This file is only for Linux pipelines that build on ubuntu. All the docker images here are based on ubuntu.
#Please don't put CentOS or manylinux2014 related stuffs here.
set -e -o -x
id
SOURCE_ROOT=$BUILD_SOURCESDIRECTORY
SCRIPT_DIR=$BUILD_SOURCESDIRECTORY/tools/ci_build/github/linux
BUILD_DIR=$BUILD_BINARIESDIRECTORY


YOCTO_VERSION="4.19"
#Training only
INSTALL_DEPS_DISTRIBUTED_SETUP=false
#Training only
ORTMODULE_BUILD=false
#Training only
USE_CONDA=false
ALLOW_RELEASED_ONNX_OPSET_ONLY_ENV="ALLOW_RELEASED_ONNX_OPSET_ONLY="$ALLOW_RELEASED_ONNX_OPSET_ONLY
echo "ALLOW_RELEASED_ONNX_OPSET_ONLY environment variable is set as "$ALLOW_RELEASED_ONNX_OPSET_ONLY_ENV

while getopts o:d:p:x:v:y:t:i:mue parameter_Option
do case "${parameter_Option}"
in
#yocto, ubuntu20.04
o) BUILD_OS=${OPTARG};;
#gpu, tensorrt or openvino. It is ignored when BUILD_OS is yocto.
d) BUILD_DEVICE=${OPTARG};;
#python version: 3.6 3.7 (absence means default 3.6)
p) PYTHON_VER=${OPTARG};;
# "--build_wheel --use_openblas"
x) BUILD_EXTR_PAR=${OPTARG};;
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
DEFAULT_PYTHON_VER="3.8"

PYTHON_VER=${PYTHON_VER:=$DEFAULT_PYTHON_VER}
echo "bo=$BUILD_OS bd=$BUILD_DEVICE bdir=$BUILD_DIR pv=$PYTHON_VER bex=$BUILD_EXTR_PAR"

GET_DOCKER_IMAGE_CMD="${SOURCE_ROOT}/tools/ci_build/get_docker_image.py"
if [[ -n "${IMAGE_CACHE_CONTAINER_REGISTRY_NAME}" ]]; then
    GET_DOCKER_IMAGE_CMD="${GET_DOCKER_IMAGE_CMD} --container-registry ${IMAGE_CACHE_CONTAINER_REGISTRY_NAME}"
fi
DOCKER_CMD="docker"


NEED_BUILD_SHARED_LIB=true
cd $SCRIPT_DIR/docker
if [ $BUILD_OS = "yocto" ]; then
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
elif [ $BUILD_DEVICE = "gpu" ]; then
        # This code path is only for training. Inferecing pipeline uses CentOS
        IMAGE="$BUILD_OS-gpu_training"
        # Current build script doesn't support building shared lib with Python dependency. To enable building with PythonOp,
        # We need to avoid `--no-undefined` when building shared lib (Otherwise, CIs will report `undefined symbols`), but removing that would bring some other concerns.
        # Plus the fact training did not need build shared library, we disable the --build_shared_lib for training CIs.
        NEED_BUILD_SHARED_LIB=false
        INSTALL_DEPS_EXTRA_ARGS="${INSTALL_DEPS_EXTRA_ARGS} -t"
        if [[ $INSTALL_DEPS_DISTRIBUTED_SETUP = true ]]; then
            INSTALL_DEPS_EXTRA_ARGS="${INSTALL_DEPS_EXTRA_ARGS} -m"
        fi
        if [[ $ORTMODULE_BUILD = true ]]; then
            INSTALL_DEPS_EXTRA_ARGS="${INSTALL_DEPS_EXTRA_ARGS} -u"
        fi
        INSTALL_DEPS_EXTRA_ARGS="${INSTALL_DEPS_EXTRA_ARGS} -v 11.8"
        $GET_DOCKER_IMAGE_CMD --repository "onnxruntime-$IMAGE" \
            --docker-build-args="--build-arg BASEIMAGE=nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-${BUILD_OS} --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER} --build-arg INSTALL_DEPS_EXTRA_ARGS=\"${INSTALL_DEPS_EXTRA_ARGS}\" --build-arg USE_CONDA=${USE_CONDA} --network=host" \
            --dockerfile Dockerfile.ubuntu_gpu_training --context .
elif [[ $BUILD_DEVICE = "tensorrt"* ]]; then
        IMAGE="$BUILD_OS-cuda11.8-cudnn8.7-tensorrt8.5"
        DOCKER_FILE=Dockerfile.ubuntu_tensorrt

        $GET_DOCKER_IMAGE_CMD --repository "onnxruntime-$IMAGE" \
            --docker-build-args="--build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER}" \
            --dockerfile $DOCKER_FILE --context .
elif [[ $BUILD_DEVICE = "openvino"* ]]; then
        BUILD_ARGS="--build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=3.8"
        IMAGE="$BUILD_OS-openvino"
        DOCKER_FILE=Dockerfile.ubuntu_openvino
        BUILD_ARGS+=" --build-arg OPENVINO_VERSION=${OPENVINO_VERSION}"
        $GET_DOCKER_IMAGE_CMD --repository "onnxruntime-$IMAGE" \
                --docker-build-args="${BUILD_ARGS}" \
                --dockerfile $DOCKER_FILE --context .
else
  exit 1
fi

if [[ $NEED_BUILD_SHARED_LIB = true ]]; then
    BUILD_EXTR_PAR=" --build_shared_lib ${BUILD_EXTR_PAR}"
fi

if [ -v EXTRA_IMAGE_TAG ]; then
    ${DOCKER_CMD} tag "onnxruntime-$IMAGE" "${EXTRA_IMAGE_TAG}"
fi

set +e
mkdir -p ~/.onnx

if [ -z "$NIGHTLY_BUILD" ]; then
    set NIGHTLY_BUILD=0
fi

if [ $BUILD_DEVICE = "cpu" ] || [ $BUILD_DEVICE = "openvino" ] || [ $BUILD_DEVICE = "arm" ]; then
    RUNTIME=
else
    RUNTIME="--gpus all"
fi

DOCKER_RUN_PARAMETER="--volume $SOURCE_ROOT:/onnxruntime_src \
                      --volume $BUILD_DIR:/build \
                      --volume /data/models:/build/models:ro \
                      --volume /data/onnx:/data/onnx:ro \
                      --volume $HOME/.cache/onnxruntime:/home/onnxruntimedev/.cache/onnxruntime \
                      --volume $HOME/.onnx:/home/onnxruntimedev/.onnx"
if [ $BUILD_DEVICE = "openvino" ] && [[ $BUILD_EXTR_PAR == *"--use_openvino GPU_FP"* ]]; then
    DOCKER_RUN_PARAMETER="$DOCKER_RUN_PARAMETER --device /dev/dri:/dev/dri"
fi
# Though this command has a yocto version argument, none of our ci build pipelines use yocto.
$DOCKER_CMD run $RUNTIME --rm $DOCKER_RUN_PARAMETER \
    -e NIGHTLY_BUILD \
    -e $ALLOW_RELEASED_ONNX_OPSET_ONLY_ENV \
    "onnxruntime-$IMAGE" \
    /bin/bash /onnxruntime_src/tools/ci_build/github/linux/run_build.sh \
    -d $BUILD_DEVICE -x "$BUILD_EXTR_PAR" -o $BUILD_OS -y $YOCTO_VERSION
