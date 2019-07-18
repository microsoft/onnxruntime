#!/bin/bash
set -e

whereis -l

whereis cmake

python3 -m pip install cmake==3.13.2

whereis cmake
cmake --version

#download Android NDK r19c
aria2c -q -d /tmp https://dl.google.com/android/repository/android-ndk-r19c-linux-x86_64.zip
unzip -oq /tmp/android-ndk-r19c-linux-x86_64.zip -d /tmp/android-ndk && mv /tmp/android-ndk/* /android-ndk
cd /
rm -rf /tmp/src

apt-get -y remove libprotobuf-dev protobuf-compiler
