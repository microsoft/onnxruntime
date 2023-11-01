#!/bin/bash
set -e

os_major_version=$(tr -dc '0-9.' < /etc/redhat-release |cut -d \. -f1)

echo "installing for os major version : $os_major_version"
dnf install -y glibc-langpack-\*
yum install -y which redhat-lsb-core expat-devel tar unzip zlib-devel make bzip2 bzip2-devel perl-IPC-Cmd openssl-devel wget

# export PATH=/opt/python/cp38-cp38/bin:$PATH

echo "installing rapidjson for AzureEP"
wget https://github.com/Tencent/rapidjson/archive/refs/tags/v1.1.0.tar.gz
tar zxvf v1.1.0.tar.gz
cd rapidjson-1.1.0
mkdir build
cd build
cmake ..
cmake --install .
cd ../..
