#!/bin/bash

while getopts p:b:i: parameter
do case "${parameter}"
in 
p) PERF_DOCKERFILE_PATH=${OPTARG};;
b) ORT_BRANCH=${OPTARG};;
i) IMAGE_NAME=${OPTARG};;
esac
done 

sudo docker build --no-cache -t $IMAGE_NAME --build-arg ONNXRUNTIME_BRANCH=$ORT_BRANCH -f $ORT_DOCKERFILE_PATH ..
