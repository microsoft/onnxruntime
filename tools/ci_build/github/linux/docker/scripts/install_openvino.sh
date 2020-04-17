#!/bin/bash
set -e
while getopts o: parameter_Option
do case "${parameter_Option}"
in
o) OPENVINO_VERSION=${OPTARG};;
esac
done

OPENVINO_VERSION=${OPENVINO_VERSION:=2020.2}
git clone https://github.com/opencv/dldt.git /data/dldt/openvino_2020.2.120

export INTEL_OPENVINO_DIR=/data/dldt/openvino_2020.2.120
apt-get update && apt-get -y  install libusb-1.0-0-dev

git checkout tags/$OPENVINO_VERSION -b $OPENVINO_VERSION
git submodule init
git submodule update --recursive
chmod +x install_dependencies.sh
./install_dependencies.sh


mkdir -p build
cd build
#cmake ..
cmake -DNGRAPH_ONNX_IMPORT_ENABLE=ON -DNGRAPH_UNIT_TEST_ENABLE=OFF -DCMAKE_BUILD_TYPE=Release ..
make --jobs=$(nproc --all)

cd ${INTEL_OPENVINO_DIR}
mkdir -p deployment_tools
mv inference-engine inference_engine && mv inference_engine deployment_tools/
ln -s deployment_tools/inference_engine inference-engine
mv model-optimizer model_optimizer && mv model_optimizer deployment_tools/
mv ngraph deployment_tools/

cd ${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine
mkdir -p lib/intel64
mkdir -p external/tbb/lib
cp -R bin/intel64/Release/lib/* lib/intel64
cp -R temp/tbb/lib/* external/tbb/lib

cd ~
