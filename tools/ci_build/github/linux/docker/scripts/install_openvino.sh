#!/bin/bash
set -e
while getopts o: parameter_Option
do case "${parameter_Option}"
in
o) OPENVINO_VERSION=${OPTARG};;
esac
done

OPENVINO_VERSION=${OPENVINO_VERSION:=2021.1}
export INTEL_OPENVINO_DIR=/data/openvino/openvino_${OPENVINO_VERSION}.110
export INTEL_OPENVINO_SRC_DIR=/data/openvino/openvino_src
git clone https://github.com/openvinotoolkit/openvino.git ${INTEL_OPENVINO_SRC_DIR}

apt-get update && apt-get -y  install libusb-1.0-0-dev

cd $INTEL_OPENVINO_SRC_DIR
git checkout tags/$OPENVINO_VERSION -b $OPENVINO_VERSION
git submodule init
git submodule update --recursive


host_cpu=$(uname -m)
sudo -E apt update
sudo -E apt-get install -y \
    build-essential \
    curl \
    wget \
    libssl-dev \
    ca-certificates \
    git \
    libboost-regex-dev \
    gcc-multilib g++-multilib \
    libgtk2.0-dev \
    pkg-config \
    unzip \
    automake \
    libtool \
    autoconf \
    libcairo2-dev \
    libpango1.0-dev \
    libglib2.0-dev \
    libgtk2.0-dev \
    libswscale-dev \
    libavcodec-dev \
    libavformat-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    libusb-1.0-0-dev \
    libopenblas-dev

if apt-cache search --names-only '^libpng12-dev'| grep -q libpng12; then
    sudo -E apt-get install -y libpng12-dev
else
    sudo -E apt-get install -y libpng-dev
fi
    
mkdir -p build
cd build

mkdir -p $INTEL_OPENVINO_DIR

cmake -DCMAKE_INSTALL_PREFIX=${INTEL_OPENVINO_DIR} -DNGRAPH_COMPONENT_PREFIX=deployment_tools/ngraph/ -DCMAKE_BUILD_TYPE=Release ..
make --jobs=$(nproc --all)
make install

cd ~
