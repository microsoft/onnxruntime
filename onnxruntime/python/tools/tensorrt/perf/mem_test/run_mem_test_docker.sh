#!/bin/bash

set -x

# Parse Arguments
while getopts d:p:l: parameter
do case "${parameter}"
in 
d) DOCKER_IMAGE=${OPTARG};; # docker image:"trt-ep-mem-test" docker image is already pre-built on perf machine
p) MEM_TEST_DIR=${OPTARG};; # mem test dir
l) BUILD_ORT_LATEST=${OPTARG};; # whether to build latest ORT
esac
done 

# Variables
DOCKER_MEM_TEST_DIR='/mem_test/'
DOCKER_ORT_LIBS='/workspace/onnxruntime/build/Linux/Release/' # This is the path on container where all ort libraries (aka libonnxruntime*.so) reside.
DOCKER_ORT_SOURCE='/workspace/onnxruntime'

if [ -z ${BUILD_ORT_LATEST} ]
then
    BUILD_ORT_LATEST="true"
fi

sudo docker run --rm --gpus all -v $MEM_TEST_DIR:$DOCKER_MEM_TEST_DIR $DOCKER_IMAGE /bin/bash $DOCKER_MEM_TEST_DIR'run.sh' -p $DOCKER_MEM_TEST_DIR -o $DOCKER_ORT_LIBS -s $DOCKER_ORT_SOURCE -l $BUILD_ORT_LATEST
