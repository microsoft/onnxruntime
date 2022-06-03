#!/bin/bash
set -e -x

mkdir -p /tmp/src

cd /tmp/src

echo "Installing cmake"
aria2c -q -d /tmp/src https://github.com/Kitware/CMake/releases/download/v3.23.2/cmake-3.23.2-linux-`uname -m`.tar.gz
tar -zxf /tmp/src/cmake-3.23.2-linux-`uname -m`.tar.gz --strip=1 -C /usr

echo "Installing Ninja"
aria2c -q -d /tmp/src https://github.com/ninja-build/ninja/archive/v1.10.0.tar.gz -o /tmp/src/ninja-linux.tar.gz
tar -zxf ninja-linux.tar.gz
cd ninja-1.10.0
cmake -Bbuild-cmake -H.
cmake --build build-cmake
mv ./build-cmake/ninja /usr/bin

echo "Installing Node.js"
CPU_ARCH=`uname -m`
if [[ "$CPU_ARCH" = "x86_64" ]]; then
  NODEJS_ARCH=x64
elif [[ "$CPU_ARCH" = "aarch64" ]]; then
  NODEJS_ARCH=arm64
else
  NODEJS_ARCH=$CPU_ARCH
fi
aria2c -q -d /tmp/src https://nodejs.org/dist/v16.14.2/node-v16.14.2-linux-${NODEJS_ARCH}.tar.gz -o /tmp/src/node-v16.14.2-linux-${NODEJS_ARCH}.tar.gz
tar --strip 1 -xf /tmp/src/node-v16.14.2-linux-${NODEJS_ARCH}.tar.gz -C /usr

cd /tmp/src
GetFile https://downloads.gradle-dn.com/distributions/gradle-6.3-bin.zip /tmp/src/gradle-6.3-bin.zip
unzip /tmp/src/gradle-6.3-bin.zip
mv /tmp/src/gradle-6.3 /usr/local/gradle

cd /
rm -rf /tmp/src
