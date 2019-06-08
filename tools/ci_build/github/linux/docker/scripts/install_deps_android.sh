#!/bin/bash
set -e
mkdir -p /tmp/src
aria2c -q -d /tmp/src https://github.com/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2.tar.gz
tar -xf /tmp/src/cmake-3.13.2.tar.gz -C /tmp/src
cd /tmp/src/cmake-3.13.2
./configure --prefix=/usr --parallel=`nproc` --system-curl --system-zlib --system-expat
make -j`nproc`
make install

#download Android NDK r19c
aria2c -q -d /tmp https://dl.google.com/android/repository/android-ndk-r19c-linux-x86_64.zip
unzip -oq /tmp/android-ndk-r19c-linux-x86_64.zip -d /tmp/android-ndk && mv /tmp/android-ndk/* /android-ndk

rm -rf /tmp/src



