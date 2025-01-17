#!/bin/bash

set -x

# Parse Arguments
while getopts w:d:p:l:c: parameter
do case "${parameter}"
in 
w) WORKSPACE=${OPTARG};; # workspace folder of onnxruntime
d) DOCKER_IMAGE=${OPTARG};; # docker image:"trt-ep-mem-test" docker image is already pre-built on perf machine
p) MEM_TEST_DIR=${OPTARG};; # mem test dir
l) BUILD_ORT_LATEST=${OPTARG};; # whether to build latest ORT
c) CONCURRENCY=${OPTARG};;
esac
done 

# Variables
DOCKER_MEM_TEST_DIR='/mem_test/'
DOCKER_ORT_SOURCE=$WORKSPACE'onnxruntime'
DOCKER_ORT_LIBS=$DOCKER_ORT_SOURCE'/build/Linux/Release/' # This is the path on container where all ort libraries (aka libonnxruntime*.so) reside.


if [ -z ${BUILD_ORT_LATEST} ]
then
    BUILD_ORT_LATEST="true"
fi

docker run --rm --gpus all -v $MEM_TEST_DIR:$DOCKER_MEM_TEST_DIR -v /data/ep-perf-models:/data/ep-perf-models $DOCKER_IMAGE /bin/bash $DOCKER_MEM_TEST_DIR'run.sh' -p $DOCKER_MEM_TEST_DIR -o $DOCKER_ORT_LIBS -s $DOCKER_ORT_SOURCE -l $BUILD_ORT_LATEST -c $CONCURRENCY
