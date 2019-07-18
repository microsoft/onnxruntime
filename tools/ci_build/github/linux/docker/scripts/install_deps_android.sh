#!/bin/bash
set -e

# cmake==3.13.2 is actually 3.12.2 lol
python3 -m pip install cmake==3.13.2.post1

cmake --version

#download Android NDK r19c
aria2c -q -d /tmp https://dl.google.com/android/repository/android-ndk-r19c-linux-x86_64.zip
unzip -oq /tmp/android-ndk-r19c-linux-x86_64.zip -d /tmp/android-ndk && mv /tmp/android-ndk/* /android-ndk
cd /
rm -rf /tmp/src

apt-get -y remove libprotobuf-dev protobuf-compiler
