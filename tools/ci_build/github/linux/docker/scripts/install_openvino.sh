#!/bin/bash
set -e
while getopts o: parameter_Option
do case "${parameter_Option}"
in
o) OPENVINO_VERSION=${OPTARG};;
esac
done

OPENVINO_VERSION=${OPENVINO_VERSION:=2018_R5}
git clone https://github.com/opencv/dldt.git /data/dldt

export INTEL_CVSDK_DIR=/data/dldt

cd ${INTEL_CVSDK_DIR}/inference-engine
git submodule init
git submodule update --recursive
git checkout tags/$OPENVINO_VERSION -b $OPENVINO_VERSION

mkdir -p build
cd build
cmake ..
make -j$(nproc)

cd ${INTEL_CVSDK_DIR}
mkdir -p deployment_tools
mv inference-engine inference_engine && mv inference_engine deployment_tools/
mv model-optimizer model_optimizer && mv model_optimizer deployment_tools/

cd ${INTEL_CVSDK_DIR}/deployment_tools/model_optimizer/install_prerequisites && ./install_prerequisites_onnx.sh

cd ${INTEL_CVSDK_DIR}/deployment_tools/inference_engine
mkdir -p lib/ubuntu_16.04/intel64
mv bin/intel64/Release/lib/* lib/ubuntu_16.04/intel64

cd ~