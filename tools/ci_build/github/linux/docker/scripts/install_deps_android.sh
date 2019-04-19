#!/bin/bash
set -e
#download Android NDK r19c
aria2c -q -d /tmp https://dl.google.com/android/repository/android-ndk-r19c-linux-x86_64.zip
unzip -oq /tmp/android-ndk-r19c-linux-x86_64.zip -d /tmp/android-ndk && mv /tmp/android-ndk/* /android-ndk
#install ninja
aria2c -q -d /tmp https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
unzip -oq /tmp/ninja-linux.zip -d /usr/bin
rm -f /tmp/ninja-linux.zip
#install protobuf
mkdir -p /tmp/src
mkdir -p /opt/cmake
aria2c -q -d /tmp/src https://github.com/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2-Linux-x86_64.tar.gz
tar -xf /tmp/src/cmake-3.13.2-Linux-x86_64.tar.gz --strip 1 -C /opt/cmake
aria2c -q -d /tmp/src https://github.com/protocolbuffers/protobuf/archive/v3.6.1.tar.gz
tar -xf /tmp/src/protobuf-3.6.1.tar.gz -C /tmp/src
cd /tmp/src/protobuf-3.6.1
if [ -f /etc/redhat-release ] ; then
  PB_LIBDIR=lib64
else
  PB_LIBDIR=lib
fi
for build_type in 'Debug' 'Relwithdebinfo'; do
  pushd .
  mkdir build_$build_type
  cd build_$build_type
  /opt/cmake/bin/cmake -G Ninja ../cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=$PB_LIBDIR  -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=$build_type
  ninja
  ninja install
  popd
done
export ONNX_ML=1

#The last onnx version will be kept

aria2c -q -d /tmp/src  http://bitbucket.org/eigen/eigen/get/3.3.7.tar.bz2
tar -jxf /tmp/src/eigen-eigen-323c052e1731.tar.bz2 -C /usr/include
mv /usr/include/eigen-eigen-323c052e1731 /usr/include/eigen3

rm -rf /tmp/src
rm -rf /usr/include/google
rm -rf /usr/lib/libproto*



