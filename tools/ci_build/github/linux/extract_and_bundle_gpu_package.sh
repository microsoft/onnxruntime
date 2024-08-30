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

mkdir -p $ARTIFACT_DIR/onnxruntime-linux-x64-tensorrt
tar zxvf $ARTIFACT_DIR/onnxruntime-linux-x64-tensorrt-*.tgz -C onnxruntime-linux-x64-tensorrt
rm $ARTIFACT_DIR/onnxruntime-linux-x64-tensorrt-*.tgz

# Rename cuda directory to gpu directory
mkdir -p $ARTIFACT_DIR/onnxruntime-linux-x64-gpu
tar zxvf $ARTIFACT_DIR/onnxruntime-linux-x64-cuda-*.tgz -C onnxruntime-linux-x64-gpu
VERSION=`ls $ARTIFACT_DIR/onnxruntime-linux-x64-gpu | sed 's/onnxruntime-linux-x64-cuda-//'`
mv $ARTIFACT_DIR/onnxruntime-linux-x64-gpu/* $ARTIFACT_DIR/onnxruntime-linux-x64-gpu/onnxruntime-linux-x64-gpu-$VERSION
rm $ARTIFACT_DIR/onnxruntime-linux-x64-cuda-*.tgz

cp onnxruntime-linux-x64-tensorrt/*/lib/libonnxruntime.so* onnxruntime-linux-x64-gpu/*/lib
cp onnxruntime-linux-x64-tensorrt/*/lib/libonnxruntime_providers_tensorrt.so onnxruntime-linux-x64-gpu/*/lib
cp onnxruntime-linux-x64-tensorrt/*/lib/libonnxruntime_providers_shared.so onnxruntime-linux-x64-gpu/*/lib
