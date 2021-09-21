#!/bin/bash
set -e -x
DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends \
        wget \
        zip \
        ca-certificates \
        curl \
        libcurl4-openssl-dev \
        libssl-dev \
        python3-dev

# Dependencies: conda
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh --no-check-certificate && /bin/bash ~/miniconda.sh -b -p /opt/miniconda
rm ~/miniconda.sh
/opt/miniconda/bin/conda clean -ya
/opt/miniconda/bin/conda install -c anaconda cmake==3.19.6 gcc_linux-64 gxx_linux-64 numpy
rm -rf /opt/miniconda/pkgs
