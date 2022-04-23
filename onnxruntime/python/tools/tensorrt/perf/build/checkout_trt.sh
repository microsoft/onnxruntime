#!/bin/bash 

echo "checking out submodule versions"

CUR_PWD=$(pwd)
cd cmake/external

# Point to correct onnx-tensorrt
echo "$1"
if [ ! "$1" = "8.2" ]
then
    cd onnx-tensorrt
    git remote update
    if [ "$1" = "8.0" ]
    then 
        git checkout "$1"'-GA'
    fi
    if [ "$1" = "7.2" ]
    then 
        git checkout "$1"'.1'
    fi 
    cd ..
fi

# Point to correct protobuf
UBUNTU_VERSION=$(lsb_release -rs)
if [$UBUNTU_VERSION = "18.04"]
then 
    cd protobuf 
    git checkout 3.10.x
    cd .. 
fi
cd $CUR_PWD 
