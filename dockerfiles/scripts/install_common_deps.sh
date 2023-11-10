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
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh --no-check-certificate && /bin/bash ~/miniconda.sh -b -p /opt/miniconda
rm ~/miniconda.sh
/opt/miniconda/bin/conda clean -ya

pip install numpy
pip install packaging
pip install "wheel>=0.35.1"
rm -rf /opt/miniconda/pkgs

# Dependencies: cmake
wget --quiet https://github.com/Kitware/CMake/releases/download/v3.26.3/cmake-3.26.3-linux-x86_64.tar.gz
tar zxf cmake-3.26.3-linux-x86_64.tar.gz
rm -rf cmake-3.26.3-linux-x86_64.tar.gz
