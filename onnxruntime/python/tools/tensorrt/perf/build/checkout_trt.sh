#!/bin/bash 

echo "$1"
if [ ! "$1" = "8.2" ]
then
    CUR_PWD=$(pwd)
    cd cmake/external/onnx-tensorrt/
    git remote update
    if [ "$1" = "8.0" ]
    then 
        echo "8.0"
        git checkout 1f416bb462689f3ef9e3f1057a113d9c6aba6972
    fi
    if [ "$1" = "7.2" ]
    then 
        ehco "7.2"
        git checkout "$1"'.1'
    fi
    cd $CUR_PWD 
fi
