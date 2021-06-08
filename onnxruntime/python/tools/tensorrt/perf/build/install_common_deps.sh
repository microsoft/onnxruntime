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
        python3-dev \
	libprotobuf-dev \
	protobuf-compiler \
	pciutils

pip install pandas coloredlogs numpy flake8 onnx Cython onnxmltools sympy packaging psutil

# Dependencies: cmake
wget --quiet https://github.com/Kitware/CMake/releases/download/v3.18.3/cmake-3.18.3-Linux-x86_64.tar.gz
tar zxf cmake-3.18.3-Linux-x86_64.tar.gz
rm -rf cmake-3.18.3-Linux-x86_64.tar.gz
