#!/bin/bash

while getopts o:p: parameter
do case "${parameter}"
in 
o) ORT_DOCKERFILE_PATH=${OPTARG};;
p) PERF_DOCKERFILE_PATH=${OPTARG};;
esac
done 

#sudo docker build -t onnxruntime-trt -f $ORT_DOCKERFILE_PATH ..
sudo docker build -t onnxruntime-trt-perf -f $PERF_DOCKERFILE_PATH ..
