#!/bin/bash
# Please run this script using run_mem_test_docker.sh
#

set -x

while getopts p:o:l:s: parameter
do case "${parameter}"
in
p) WORKSPACE=${OPTARG};;
o) ORT_BINARY_PATH=${OPTARG};;
l) BUILD_ORT_LATEST=${OPTARG};;
s) ORT_SOURCE=${OPTARG};;
esac
done

ONNX_MODEL_URL="https://github.com/onnx/models/raw/master/vision/classification/squeezenet/model/squeezenet1.0-7.onnx"
ONNX_MODEL="squeezenet.onnx"
ASAN_OPTIONS="protect_shadow_gap=0:log_path=asan.log"

export LD_LIBRARY_PATH=${ORT_BINARY_PATH}
export LIBRARY_PATH=${ORT_BINARY_PATH}

if [ -z ${BUILD_ORT_LATEST} ]
then
    BUILD_ORT_LATEST="true"
fi

if [ -z ${ORT_SOURCE} ]
then
    ORT_SOURCE="/code/onnxruntime/"
fi

if [ ${BUILD_ORT_LATEST} == "true" ]
then
    cd ${ORT_SOURCE}
    git pull
    ./build.sh --parallel --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --use_tensorrt --tensorrt_home /usr/lib/x86_64-linux-gnu/ \
               --config Release --build_shared_lib --update --build --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER)
fi

cd ${WORKSPACE}

mkdir build
cd build
cp ../squeezenet_calibration.flatbuffers . 

cmake ..
make -j8
wget ${ONNX_MODEL_URL} -O ${ONNX_MODEL}
ASAN_OPTIONS=${ASAN_OPTIONS} ./onnx_memtest

if [ $? -ne 0 ]
then
    echo "Memory test application failed."
    exit 1
fi

mkdir result
if [ -e asan.log* ]
then
    cat asan.log*
    mv asan.log* result
else
    echo "No memory Leak(s) or other memory error(s) detected." > result/asan.log
fi
