#!/bin/bash

while getopts p:b:i:t:c: parameter
do case "${parameter}"
in 
p) PERF_DOCKERFILE_PATH=${OPTARG};;
b) ORT_BRANCH=${OPTARG};;
i) IMAGE_NAME=${OPTARG};;
t) TRT=${OPTARG};;
c) CUDA_VERSION=${OPTARG};;
esac
done 

sudo docker build --no-cache -t $IMAGE_NAME --build-arg TRT=$TRT --build-arg CUDA_VERSION=$CUDA_VERSION --build-arg ONNXRUNTIME_BRANCH=$ORT_BRANCH -f $PERF_DOCKERFILE_PATH .
