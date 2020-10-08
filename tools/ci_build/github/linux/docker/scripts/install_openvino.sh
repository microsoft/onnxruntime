#!/bin/bash
set -e
while getopts o: parameter_Option
do case "${parameter_Option}"
in
o) OPENVINO_VERSION=${OPTARG};;
esac
done

OPENVINO_VERSION=${OPENVINO_VERSION:=2020.4}
export INTEL_OPENVINO_DIR=/data/openvino/openvino_${OPENVINO_VERSION}.287
export INTEL_OPENVINO_SRC_DIR=/data/openvino/openvino_src
git clone https://github.com/openvinotoolkit/openvino.git ${INTEL_OPENVINO_SRC_DIR}

apt-get update && apt-get -y  install libusb-1.0-0-dev

cd $INTEL_OPENVINO_SRC_DIR
git checkout tags/$OPENVINO_VERSION -b $OPENVINO_VERSION
git submodule init
git submodule update --recursive
chmod +x install_dependencies.sh
./install_dependencies.sh


mkdir -p build
cd build

mkdir -p $INTEL_OPENVINO_DIR

cmake -DCMAKE_INSTALL_PREFIX=${INTEL_OPENVINO_DIR} -DNGRAPH_COMPONENT_PREFIX=deployment_tools/ngraph/ -DCMAKE_BUILD_TYPE=Release ..
make --jobs=$(nproc --all)
make install

cd ~
