#!/bin/bash

set -x
# Parse Arguments
while getopts d:p: parameter
do case "${parameter}"
in 
d) DOCKER_IMAGE=${OPTARG};; # the "trt-ep-mem-test" docker image is already pre-built on perf machine.
p) MEM_TEST_DIR=${OPTARG};;
esac
done 

# Variables
DOCKER_MEM_TEST_DIR='/mem_test/'
DOCKER_ORT_LIBS='/workspace/onnxruntime/build/Linux/Release/' # This is the path where all ort libraries (aka libonnxruntime*.so) reside.

sudo docker run --gpus all -v $MEM_TEST_DIR:$DOCKER_MEM_TEST_DIR $DOCKER_IMAGE /bin/bash $DOCKER_MEM_TEST_DIR'run.sh' -p $DOCKER_MEM_TEST_DIR -o $DOCKER_ORT_LIBS
