#!/bin/bash 

echo "checking out submodule versions"

CUR_PWD=$(pwd)
cd onnxruntime/cmake/external


echo "$1"
if [ ! "$1" = "8.2" ]
then
    # Point to correct onnx-tensorrt
    cd onnx-tensorrt
    git remote update
    if [ "$1" = "8.0" ]
    then 
        git checkout "$1"'-GA'
    fi
    cd ..

    # checkout correct protobuf to match onnx-tensorrt
    cd protobuf 
    git checkout 3.10.x
    cd .. 

fi

cd $CUR_PWD 