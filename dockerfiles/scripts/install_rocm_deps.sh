#!/bin/bash
prefix=/opt/rocm
DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends \
        wget \
        zip \
        ca-certificates \
        build-essential \
        curl \
        libcurl4-openssl-dev \
        libssl-dev \
        python3-dev

# rocm-cmake
wget --quiet https://github.com/RadeonOpenCompute/rocm-cmake/archive/rocm-3.8.0.tar.gz
tar -xzvf rocm-3.8.0.tar.gz
rm rocm-3.8.0.tar.gz
cd rocm-cmake-rocm-3.8.0
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$prefix ..
make -j8
make install
cd ../..
rm -rf rocm-cmake-rocm-3.8.0

# rccl
wget --quiet https://github.com/ROCmSoftwarePlatform/rccl/archive/rocm-4.0.0.tar.gz
tar -xzvf rocm-4.0.0.tar.gz
rm rocm-4.0.0.tar.gz
cd rccl-rocm-4.0.0
mkdir build
cd build
CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_INSTALL_PREFIX=$prefix ..
make -j8
make install
cd ../..
rm -rf rccl-rocm-4.0.0

#rocrand
wget --quiet https://github.com/ROCmSoftwarePlatform/rocRAND/archive/rocm-4.0.0.tar.gz
tar -xzvf rocm-4.0.0.tar.gz
rm rocm-4.0.0.tar.gz
cd rocRAND-rocm-4.0.0
mkdir build
cd build
CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_INSTALL_PREFIX=$prefix ..
make -j8
make install
cd ../..
rm -rf rocRAND-rocm-4.0.0

#hipcub
wget --quiet https://github.com/ROCmSoftwarePlatform/hipCUB/archive/rocm-4.0.0.tar.gz
tar -xzvf rocm-4.0.0.tar.gz
rm rocm-4.0.0.tar.gz
cd hipCUB-rocm-4.0.0
mkdir build
cd build
CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_INSTALL_PREFIX=$prefix ..
make -j8
make package
make install
cd ../..
rm -rf hipCUB-rocm-4.0.0

#rocprim
wget --quiet https://github.com/ROCmSoftwarePlatform/rocPRIM/archive/rocm-4.0.0.tar.gz
tar -xzvf rocm-4.0.0.tar.gz
rm rocm-4.0.0.tar.gz
cd rocPRIM-rocm-4.0.0
mkdir build
cd build
CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_INSTALL_PREFIX=$prefix ..
make -j8
make install
cd ../..
rm -rf rocPRIM-rocm-4.0.0

