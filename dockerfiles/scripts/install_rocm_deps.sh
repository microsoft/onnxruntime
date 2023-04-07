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
rocm_cmake_version=4.5.2
wget --quiet https://github.com/RadeonOpenCompute/rocm-cmake/archive/refs/tags/rocm-${rocm_cmake_version}.tar.gz
tar -xzvf rocm-${rocm_cmake_version}.tar.gz
rm rocm-${rocm_cmake_version}.tar.gz
cd rocm-cmake-rocm-${rocm_cmake_version}
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$prefix ..
make -j8
make install
cd ../..
rm -rf rocm-cmake-rocm-${rocm_cmake_version}

# rccl
rccl_version=4.5.2
wget --quiet https://github.com/ROCmSoftwarePlatform/rccl/archive/refs/tags/rocm-${rccl_version}.tar.gz
tar -xzvf rocm-${rccl_version}.tar.gz
rm rocm-${rccl_version}.tar.gz
cd rccl-rocm-${rccl_version}
mkdir build
cd build
CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_INSTALL_PREFIX=$prefix ..
make -j8
make install
cd ../..
rm -rf rccl-rocm-${rccl_version}

#rocrand
rocrand_version=4.5.2
wget --quiet https://github.com/ROCmSoftwarePlatform/rocRAND/archive/refs/tags/rocm-${rocrand_version}.tar.gz
tar -xzvf rocm-${rocrand_version}.tar.gz
rm rocm-${rocrand_version}.tar.gz
cd rocRAND-rocm-${rocrand_version}
mkdir build
cd build
CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_INSTALL_PREFIX=$prefix ..
make -j8
make install
cd ../..
rm -rf rocRAND-rocm-${rocrand_version}

#hipcub
hipcub_version=4.5.2
wget --quiet https://github.com/ROCmSoftwarePlatform/hipCUB/archive/refs/tags/rocm-${hipcub_version}.tar.gz
tar -xzvf rocm-${hipcub_version}.tar.gz
rm rocm-${hipcub_version}.tar.gz
cd hipCUB-rocm-${hipcub_version}
mkdir build
cd build
CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_INSTALL_PREFIX=$prefix ..
make -j8
make package
make install
cd ../..
rm -rf hipCUB-rocm-${hipcub_version}

#rocprim
rocprim_version=4.5.2
wget --quiet https://github.com/ROCmSoftwarePlatform/rocPRIM/archive/refs/tags/rocm-${rocprim_version}.tar.gz
tar -xzvf rocm-${rocprim_version}.tar.gz
rm rocm-${rocprim_version}.tar.gz
cd rocPRIM-rocm-${rocprim_version}
mkdir build
cd build
CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_INSTALL_PREFIX=$prefix ..
make -j8
make install
cd ../..
rm -rf rocPRIM-rocm-${rocprim_version}

