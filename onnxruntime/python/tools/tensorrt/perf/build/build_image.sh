#!/bin/bash

while getopts o:p:b:i:t:c: parameter
do case "${parameter}"
in 
o) TRT_DOCKERFILE_PATH=${OPTARG};;
p) PERF_DOCKERFILE_PATH=${OPTARG};;
b) ORT_BRANCH=${OPTARG};;
i) IMAGE_NAME=${OPTARG};;
t) TRT_VERSION=${OPTARG};;
c) CMAKE_CUDA_ARCHITECTURES=${OPTARG};;
esac
done 

IMAGE=onnxruntime

docker build --no-cache -t $IMAGE --build-arg CMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES --build-arg TRT_VERSION=$TRT_VERSION --build-arg ONNXRUNTIME_BRANCH=$ORT_BRANCH -f $TRT_DOCKERFILE_PATH . 
docker build --no-cache --build-arg IMAGE=$IMAGE --build-arg CMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES --build-arg ONNXRUNTIME_BRANCH=$ORT_BRANCH --build-arg TRT_VERSION=$TRT_VERSION -t $IMAGE_NAME -f $PERF_DOCKERFILE_PATH .
