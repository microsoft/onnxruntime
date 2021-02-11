#!/bin/bash

while getopts o:p:b:i: parameter
do case "${parameter}"
in 
o) ORT_DOCKERFILE_PATH=${OPTARG};;
p) PERF_DOCKERFILE_PATH=${OPTARG};;
b) ORT_BRANCH=${OPTARG};;
i) IMAGE_NAME=${OPTARG};;
esac
done 

sudo docker build -t onnxruntime-trt --build-arg ONNXRUNTIME_BRANCH=$ORT_BRANCH -f $ORT_DOCKERFILE_PATH ..
echo "sudo docker build -t $IMAGE_NAME -f $PERF_DOCKERFILE_PATH .."
sudo docker build -t $IMAGE_NAME -f $PERF_DOCKERFILE_PATH ..
