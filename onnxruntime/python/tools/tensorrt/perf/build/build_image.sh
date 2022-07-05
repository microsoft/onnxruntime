#!/bin/bash

while getopts o:b:i:t:c: parameter
do case "${parameter}"
in 
o) TRT_DOCKERFILE_PATH=${OPTARG};;
b) ORT_BRANCH=${OPTARG};;
i) IMAGE_NAME=${OPTARG};;
t) TRT_CONTAINER=${OPTARG};;
c) CMAKE_CUDA_ARCHITECTURES=${OPTARG};;
esac
done 

docker build --no-cache -t $IMAGE_NAME --build-arg CMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES --build-arg TRT_CONTAINER_VERSION=$TRT_CONTAINER --build-arg ONNXRUNTIME_BRANCH=$ORT_BRANCH -f $TRT_DOCKERFILE_PATH . 