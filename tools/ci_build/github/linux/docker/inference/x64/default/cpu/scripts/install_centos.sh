#!/bin/bash
set -e -x

os_major_version=$(cat /etc/redhat-release | tr -dc '0-9.'|cut -d \. -f1)
CPU_ARCH=`uname -m`
PACKAGES_TO_INSTALL="which gdb redhat-lsb-core expat-devel tar unzip zlib-devel make libunwind  aria2  bzip2 bzip2-devel java-11-openjdk-devel graphviz devtoolset-11-binutils devtoolset-11-gcc devtoolset-11-gcc-c++ devtoolset-11-gcc-gfortran python3 python3-devel python3-pip"

echo "installing for os major version : $os_major_version"
yum install -y centos-release-scl-rh
yum install -y $PACKAGES_TO_INSTALL

pip3 install --upgrade pip
localedef -i en_US -f UTF-8 en_US.UTF-8
