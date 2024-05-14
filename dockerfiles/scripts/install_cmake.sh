#!/bin/bash
set -e -x
mkdir -p /tmp/src
cd /tmp/src

echo "Installing cmake"
CPU_ARCH=`uname -m`
CMAKE_VERSION='3.27.3'
curl https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-linux-$CPU_ARCH.tar.gz -sSL --retry 5  -o /tmp/src/cmake.tar.gz
tar -zxf /tmp/src/cmake.tar.gz --strip=1 -C /usr
rm -f /tmp/src/cmake.tar.gz
