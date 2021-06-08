#!/bin/bash
set -e

python3 -m pip install cmake==3.20.2

cmake --version

# Download Android NDK r20, move /temp_ndk/android-ndk-<version> to /android_ndk
wget -qO- -O temp.zip https://dl.google.com/android/repository/android-ndk-r20-linux-x86_64.zip && unzip -oq temp.zip -d /temp-ndk && mv /temp-ndk/* /android-ndk && rm temp.zip && rm -rf /temp-ndk && ls /android-ndk

apt-get -y remove libprotobuf-dev protobuf-compiler
