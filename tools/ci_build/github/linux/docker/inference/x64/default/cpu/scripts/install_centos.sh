#!/bin/bash
set -e -x

os_major_version=$(cat /etc/redhat-release | tr -dc '0-9.'|cut -d \. -f1)

echo "installing for CentOS version : $os_major_version"
if [[ "$os_major_version" = "7" ]]; then
  PACKAGE_MANAGER=yum
  GCC_VERSION=11
elif [[ "$os_major_version" = "8" ]]; then
  PACKAGE_MANAGER=dnf
  GCC_VERSION=12
fi

$PACKAGE_MANAGER install -y glibc-langpack-\* glibc-locale-source which gdb redhat-lsb-core expat-devel tar unzip zlib-devel make bzip2 bzip2-devel java-11-openjdk-devel graphviz gcc-toolset-$GCC_VERSION-binutils gcc-toolset-$GCC_VERSION-gcc gcc-toolset-$GCC_VERSION-gcc-c++ gcc-toolset-$GCC_VERSION-gcc-gfortran
locale