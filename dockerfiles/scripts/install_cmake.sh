#!/bin/bash
mkdir -p /tmp/src
cd /tmp/src

echo "Installing cmake"
curl https://github.com/Kitware/CMake/releases/download/v3.27.2/cmake-3.27.2-linux-`uname -m`.tar.gz -sSL --retry 5  -o /tmp/src/cmake-3.27.2-linux-`uname -m`.tar.gz
tar -zxf /tmp/src/cmake-3.27.2-linux-`uname -m`.tar.gz --strip=1 -C /usr
