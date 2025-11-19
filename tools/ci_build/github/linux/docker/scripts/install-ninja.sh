#!/bin/bash
set -e -x
mkdir -p /tmp/ninja 
cd /tmp/ninja 
curl -O -sSL https://github.com/ninja-build/ninja/archive/v1.12.1.tar.gz 
tar -zxvf v1.12.1.tar.gz --strip=1 
cmake -Bbuild-cmake -H. 
cmake --build build-cmake 
mv ./build-cmake/ninja /usr/bin 
cd / 
rm -rf /tmp/ninja
