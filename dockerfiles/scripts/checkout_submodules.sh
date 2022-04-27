#!/bin/bash 

echo "checking out submodule versions"

downgrade_protobuf=false
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
        downgrade_protobuf=true
    fi
    cd ..

    # checkout correct protobuf to match onnx-tensorrt
    if [ "$downgrade_protobuf" = true ]
    then
        cd protobuf 
        git checkout 3.10.x
        cd .. 
    fi

fi

cd "$CUR_PWD" 
