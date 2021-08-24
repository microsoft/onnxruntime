#!/bin/bash
set -e -o -x

while getopts a: parameter_Option
do case "${parameter_Option}"
in
a) ARTIFACT_DIR=${OPTARG};;
esac
done

EXIT_CODE=1

uname -a

cd $ARTIFACT_DIR 

mkdir -p $ARTIFACT_DIR/onnxruntime-linux-x64-gpu-tensorrt
tar zxvf $ARTIFACT_DIR/onnxruntime-linux-x64-gpu-tensorrt-*.tgz -C onnxruntime-linux-x64-gpu-tensorrt
rm $ARTIFACT_DIR/onnxruntime-linux-x64-gpu-tensorrt-*.tgz

mkdir -p $ARTIFACT_DIR/onnxruntime-linux-x64-gpu
TAR_NAME=`ls $ARTIFACT_DIR/onnxruntime-linux-x64-gpu-*.tgz`
tar zxvf $ARTIFACT_DIR/onnxruntime-linux-x64-gpu-*.tgz -C onnxruntime-linux-x64-gpu
rm $ARTIFACT_DIR/onnxruntime-linux-x64-gpu-*.tgz

cp onnxruntime-linux-x64-gpu-tensorrt/*/lib/libonnxruntime.so* onnxruntime-linux-x64-gpu/*/lib
cp onnxruntime-linux-x64-gpu-tensorrt/*/lib/libonnxruntime_providers_tensorrt.so onnxruntime-linux-x64-gpu/*/lib
cp onnxruntime-linux-x64-gpu-tensorrt/*/lib/libonnxruntime_providers_shared.so onnxruntime-linux-x64-gpu/*/lib
