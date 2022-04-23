#!/bin/bash 

echo "checking out onnx-tensorrt for correct version"
echo "$1"
if [ ! "$1" = "8.2" ]
then
    CUR_PWD=$(pwd)
    cd cmake/external
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
    cd protobuf 
    git checkout 3.10.x
    cd .. 
    cd $CUR_PWD 
fi

