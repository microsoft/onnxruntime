#!/bin/bash

while getopts p:b:i:t: parameter
do case "${parameter}"
in 
p) PERF_DOCKERFILE_PATH=${OPTARG};;
b) ORT_BRANCH=${OPTARG};;
i) IMAGE_NAME=${OPTARG};;
t) TRT_CONTAINER=${OPTARG};;
esac
done 

sudo docker build --no-cache -t $IMAGE_NAME --build-arg TRT_CONTAINER=$TRT_CONTAINER --build-arg ONNXRUNTIME_BRANCH=$ORT_BRANCH -f $PERF_DOCKERFILE_PATH .
