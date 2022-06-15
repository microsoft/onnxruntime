#!/bin/bash
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

# Dependencies: conda
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh --no-check-certificate && /bin/bash ~/miniconda.sh -b -p /opt/miniconda
rm ~/miniconda.sh
conda clean -ya

conda install numpy

# Dependencies: cmake
wget --quiet https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0-linux-x86_64.tar.gz
tar zxf cmake-3.21.0-linux-x86_64.tar.gz
rm -rf cmake-3.21.0-linux-x86_64.tar.gz
