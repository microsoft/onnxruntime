#!/bin/bash
set -e -x

os_major_version=$(cat /etc/redhat-release | tr -dc '0-9.'|cut -d \. -f1)

echo "installing for CentOS version : $os_major_version"
yum install -y centos-release-scl-rh
yum install -y which gdb redhat-lsb-core expat-devel tar unzip zlib-devel make libunwind bzip2 bzip2-devel java-11-openjdk-devel graphviz devtoolset-11-binutils devtoolset-11-gcc devtoolset-11-gcc-c++ devtoolset-11-gcc-gfortran rh-python38-python rh-python38-python-pip

/opt/rh/rh-python38/root/usr/bin/python3.8 -m pip install --upgrade pip

# enable Python 3.8 by default
echo "source scl_source enable rh-python38" > /etc/profile.d/enablepython38.sh

localedef -i en_US -f UTF-8 en_US.UTF-8
