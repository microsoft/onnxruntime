#!/bin/bash 

echo "checking out submodule versions"

downgrade_protobuf=false
CUR_PWD=$(pwd)
cd onnxruntime/cmake/external || return

echo "$1"
if [ ! "$1" = "8.2" ]
then
    # Point to correct onnx-tensorrt
    downgrade_protobuf=true
    ( 
    cd onnx-tensorrt || return
    git remote update
    if [ "$1" = "8.0" ]
    then 
        git checkout "$1"'-GA'
    fi 
    )

    # checkout correct protobuf to match onnx-tensorrt
    if [ "$downgrade_protobuf" = true ]
    then
        ( 
        cd protobuf || return
        git checkout 3.10.x 
        )
    fi

fi

cd "$CUR_PWD" || return
