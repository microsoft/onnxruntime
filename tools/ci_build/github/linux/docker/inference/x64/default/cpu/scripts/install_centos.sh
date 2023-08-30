#!/bin/bash
set -e -x

os_major_version=$(cat /etc/redhat-release | tr -dc '0-9.'|cut -d \. -f1)

echo "installing for CentOS version : $os_major_version"

dnf install -y glibc-langpack-\* glibc-locale-source which gdb redhat-lsb-core expat-devel tar unzip zlib-devel make bzip2 bzip2-devel java-11-openjdk-devel graphviz gcc-toolset-12-binutils gcc-toolset-12-gcc gcc-toolset-12-gcc-c++ gcc-toolset-12-gcc-gfortran
locale