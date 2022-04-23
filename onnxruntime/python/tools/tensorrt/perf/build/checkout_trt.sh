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

# deprecate
#UBUNTU_VERSION=$(lsb_release -rs)
#echo $UBUNTU_VERSION
#if [ $UBUNTU_VERSION = "18.04" ]

PROTO_VER=$(apt list | grep -oP -m 1 'protobuf-compiler.* \K[\d\.]+')
echo "$PROTO_VER"
if [[ $PROTO_VER < "3.11.0" ]]
then 
    cd protobuf 
    git checkout $(echo $PROTO_VER | grep -oP -m 1 '\d+\.\d+\.')x
    cd .. 
fi
cd $CUR_PWD 
