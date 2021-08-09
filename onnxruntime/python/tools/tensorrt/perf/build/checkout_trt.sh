#!/bin/bash 

if [ ! "$1" = "8.0" ]
then
    CUR_PWD=$(pwd)
    cd cmake/external/onnx-tensorrt/
    git remote update
    if [ "$1" = "7.1" ]
    then 
        git checkout "$1"
    else 
        git checkout "$1"'.1'
    fi
    cd $CUR_PWD 
fi
