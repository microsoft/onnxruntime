#!/bin/bash

while getopts o:p:b:i:t: parameter
do case "${parameter}"
in 
o) ORT_TRT_DOCKERFILE_PATH=${OPTARG};;
p) PERF_DOCKERFILE_PATH=${OPTARG};;
b) ORT_BRANCH=${OPTARG};;
i) IMAGE_NAME=${OPTARG};;
t) TRT=${OPTARG};;
esac
done 

IMAGE=onnxruntime 

docker build --no-cache -t $IMAGE --build-arg TRT=$TRT --build-arg ONNXRUNTIME_BRANCH=$ORT_BRANCH -f $ORT_TRT_DOCKERFILE_PATH . 
<<<<<<< HEAD
docker build --no-cache --build-arg IMAGE=$IMAGE --build-arg ONNXRUNTIME_BRANCH=$ORT_BRANCH --build-arg TRT_VERSION=$TRT -t $IMAGE_NAME -f $PERF_DOCKERFILE_PATH .
=======
docker build --no-cache --build-arg IMAGE=$IMAGE -t $IMAGE_NAME -f $PERF_DOCKERFILE_PATH .
>>>>>>> 6ecf626a9c491caffe3bd481d638b27d14534a6f
