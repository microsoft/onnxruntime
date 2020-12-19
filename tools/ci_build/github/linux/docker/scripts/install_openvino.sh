#!/bin/bash
set -e
while getopts o: parameter_Option
do case "${parameter_Option}"
in
o) OPENVINO_VERSION=${OPTARG};;
esac
done

OPENVINO_VERSION=${OPENVINO_VERSION:=2021.2}
export INTEL_OPENVINO_DIR=/opt/intel/openvino_${OPENVINO_VERSION}.185

mkdir /data && cd /data
apt-get update && apt-get -y  install libusb-1.0-0-dev
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
git remote prune origin
git fetch --all
git checkout tags/$OPENVINO_VERSION -b $OPENVINO_VERSION
git submodule init
git submodule update --recursive
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INTEL_OPENVINO_DIR -DNGRAPH_COMPONENT_PREFIX=deployment_tools/ngraph/
make --jobs=$(nproc --all)
make install

cd ~