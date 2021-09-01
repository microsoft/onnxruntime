#!/bin/bash

while getopts o:p:b:i:t: parameter
do case "${parameter}"
in 
o) TRT_DOCKERFILE_PATH=${OPTARG};;
p) PERF_DOCKERFILE_PATH=${OPTARG};;
b) ORT_BRANCH=${OPTARG};;
i) IMAGE_NAME=${OPTARG};;
t) TRT_VERSION=${OPTARG};;
esac
done 

IMAGE=onnxruntime
CMAKE_CUDA_ARCHITECTURES=75 

docker build --no-cache -t $IMAGE --build-arg CMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES --build-arg TRT_VERSION=$TRT_VERSION --build-arg ONNXRUNTIME_BRANCH=$ORT_BRANCH -f $TRT_DOCKERFILE_PATH . 
docker build --no-cache --build-arg IMAGE=$IMAGE --build-arg CMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES --build-arg ONNXRUNTIME_BRANCH=$ORT_BRANCH --build-arg TRT_VERSION=$TRT_VERSION -t $IMAGE_NAME -f $PERF_DOCKERFILE_PATH .
