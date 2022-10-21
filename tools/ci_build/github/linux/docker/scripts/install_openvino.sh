#!/bin/bash
set -e
while getopts o: parameter_Option
do case "${parameter_Option}"
in
o) OPENVINO_VERSION=${OPTARG};;
esac
done

OPENVINO_VERSION=${OPENVINO_VERSION:=2019_R1.1}
git clone --branch releases/2019/3 https://github.com/opencv/dldt.git /data/dldt/openvino_2019.1.144

export INTEL_CVSDK_DIR=/data/dldt/openvino_2019.1.144
apt-get update && apt-get -y  install libusb-1.0-0-dev

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
mkdir -p lib/intel64
mkdir -p external/tbb/lib
mv bin/intel64/Release/lib/* lib/intel64
mv temp/tbb/lib/* external/tbb/lib

cd ~
