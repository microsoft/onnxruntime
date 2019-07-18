#!/bin/bash
set -e

# cmake==3.13.2 is actually 3.12.2 lol
python3 -m pip install cmake==3.13.2.post1

cmake --version

# Download Android SDK Manager
wget -qO- -O temp.zip https://dl.google.com/android/repository/sdk-tools-linux-4333796.zip && unzip -oq temp.zip -d /android-sdk && rm temp.zip

mkdir /android-sdk/platforms /android-sdk/platform-tools

# Download Android NDK r19c
wget -qO- -O temp.zip https://dl.google.com/android/repository/android-ndk-r19c-linux-x86_64.zip && unzip -oq temp.zip -d /android-ndk && rm temp.zip

apt-get -y remove libprotobuf-dev protobuf-compiler
