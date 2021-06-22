#!/bin/bash
set -x

while getopts p: parameter
do case "${parameter}"
in
p) ORT_BINARY_PATH=${OPTARG};;
esac
done

ONNX_MODEL_URL="https://github.com/onnx/models/raw/master/vision/classification/squeezenet/model/squeezenet1.0-7.onnx"
ONNX_MODEL="squeezenet.onnx"
ASAN_OPTIONS="protect_shadow_gap=0:log_path=asan.log"

export LD_LIBRARY_PATH=$ORT_BINARY_PATH
export LIBRARY_PATH=$ORT_BINARY_PATH

mkdir build
cd build
cp ../squeezenet_calibration.flatbuffers . 

cmake ..
make -j8
wget $ONNX_MODEL_URL -O $ONNX_MODEL
ASAN_OPTIONS=$ASAN_OPTIONS ./onnx_memtest

if [ -e asan.log* ]
then
    cat asan.log*
else
    echo "No memory Leak(s) or other memory error(s) detected."
fi
