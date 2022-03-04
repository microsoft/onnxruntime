#!/bin/bash
set -x

while getopts w:p: parameter
do case "${parameter}"
in
p) ORT_PACKAGE=${OPTARG};;
w) WORKSPACE=${OPTARG};;
esac
done

#ONNX_MODEL_URL="https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.0-7.onnx"
ONNX_MODEL_URL="https://media.githubusercontent.com/media/onnx/models/main/vision/classification/squeezenet/model/squeezenet1.0-7.onnx"
ONNX_MODEL="squeezenet.onnx"

CUR_PWD=$(pwd)
cd ${WORKSPACE}

tar zxvf ${ORT_PACKAGE}
ORT_LIB="${ORT_PACKAGE%.*}/lib"

export LD_LIBRARY_PATH=${ORT_LIB}
export LIBRARY_PATH=${ORT_LIB}

mkdir -p build
cd build

cmake ..
make -j4
#wget ${ONNX_MODEL_URL} -O ${ONNX_MODEL}
curl ${ONNX_MODEL_URL} --output ${ONNX_MODEL}
./capi_test

if [ $? -ne 0 ]
then
    echo "capi test application failed."
    cd ${CUR_PWD}
    exit 1
fi

cd ${CUR_PWD}
