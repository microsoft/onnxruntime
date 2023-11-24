#!/bin/bash
set -e -x

os_major_version=$(tr -dc '0-9.' < /etc/redhat-release |cut -d \. -f1)

echo "installing for CentOS version : $os_major_version"
dnf install -y python39-devel glibc-langpack-\* glibc-locale-source which redhat-lsb-core expat-devel tar unzip zlib-devel make bzip2 bzip2-devel java-11-openjdk-devel graphviz gcc-toolset-12-binutils gcc-toolset-12-gcc gcc-toolset-12-gcc-c++ gcc-toolset-12-gcc-gfortran
locale
