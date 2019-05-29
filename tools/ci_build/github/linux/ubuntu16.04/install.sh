#!/bin/bash
set -e
DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends \
        autotools-dev \
        build-essential \
        git ca-certificates \
        ca-certificates \
        pkg-config \
        wget \
        zlib1g \
        zlib1g-dev \
        libssl-dev \
        curl \
        autoconf \
        sudo \
        gfortran \
        python3-dev \
        language-pack-en \
        libopenblas-dev \
        liblttng-ust0 \
        libcurl3 \
        libssl1.0.0 \
        libkrb5-3 \
        libicu55 \
        aria2 \
        bzip2 \
        unzip \
        rsync libunwind8 \
        python3-setuptools python3-numpy python3-wheel python python3-pip

locale-gen en_US.UTF-8
update-locale LANG=en_US.UTF-8

rm -rf /var/lib/apt/lists/*

aria2c -q -d /tmp https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
unzip -q /tmp/ninja-linux.zip -d /usr/bin
mkdir -p /tmp/azcopy
aria2c -q -d /tmp/azcopy -o azcopy.tar.gz https://aka.ms/downloadazcopylinux64
tar -xf /tmp/azcopy/azcopy.tar.gz -C /tmp/azcopy
/tmp/azcopy/install.sh
rm -rf /tmp/azcopy
#install protobuf
mkdir -p /tmp/src
mkdir -p /opt/cmake
aria2c -q -d /tmp/src   https://github.com/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2-Linux-x86_64.tar.gz
tar -xf /tmp/src/cmake-3.13.2-Linux-x86_64.tar.gz --strip 1 -C /opt/cmake
aria2c -q -d /tmp/src https://github.com/protocolbuffers/protobuf/archive/v3.6.1.tar.gz
tar -xf /tmp/src/protobuf-3.6.1.tar.gz -C /tmp/src
cd /tmp/src/protobuf-3.6.1
for build_type in 'Debug' 'Relwithdebinfo'; do
  pushd .
  mkdir build_$build_type
  cd build_$build_type
  cmake -G Ninja ../cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=lib  -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=$build_type
  ninja
  ninja install
  popd
done
export ONNX_ML=1
#3376d4438aaadfba483399fa249b841153152bc0 is v1.2.2
for onnx_version in "3376d4438aaadfba483399fa249b841153152bc0" "6f91908b6a894278377e2767dc9ce75ce197fb88" ; do
  aria2c -q -d /tmp/src  https://github.com/onnx/onnx/archive/$onnx_version.tar.gz
  tar -xf /tmp/src/onnx-$onnx_version.tar.gz -C /tmp/src
  cd /tmp/src/onnx-$onnx_version
  git clone https://github.com/pybind/pybind11.git third_party/pybind11
  python3 setup.py bdist_wheel
  pip3 install -q dist/*
  mkdir -p /data/onnx/$onnx_version
  backend-test-tools generate-data -o /data/onnx/$onnx_version
  pip3 uninstall -y onnx
done

chmod 0777 /data/onnx
rm -rf /tmp/src


