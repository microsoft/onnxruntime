#!/bin/bash
set -e

os_major_version=$(cat /etc/redhat-release | tr -dc '0-9.'|cut -d \. -f1)

echo "installing for os major version : $os_major_version"
yum install -y which gdb redhat-lsb-core expat-devel tar unzip zlib-devel make libunwind bzip2 bzip2-devel perl-IPC-Cmd python3 openssl-devel wget

echo "installing rapidjson for AzureEP"
wget https://github.com/Tencent/rapidjson/archive/refs/tags/v1.1.0.tar.gz
tar zxvf v1.1.0.tar.gz
cd rapidjson-1.1.0
mkdir build
cd build
echo "prepare to install rapidjson"
cmake ..
cmake --install .
cd ../..
echo "rapidjson installed"