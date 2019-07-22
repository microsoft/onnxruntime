#!/bin/bash
set -e

apt update && apt install -y openjdk-8-jdk

# cmake==3.13.2 is actually 3.12.2 lol
python3 -m pip install cmake==3.13.2.post1

cmake --version

# Download Android SDK Manager
wget -qO- -O temp.zip https://dl.google.com/android/repository/sdk-tools-linux-4333796.zip && unzip -oq temp.zip -d /android-sdk && rm temp.zip

echo "y" | /android-sdk/tools/bin/sdkmanager --install 'system-images;android-28;google_apis;x86_64'
echo "y" | /android-sdk/tools/bin/sdkmanager --install 'platform-tools'

chmod a+x /android-sdk/tools/bin/avdmanager
chmod a+x /android-sdk/emulator/emulator
chmod a+x /android-sdk/platform-tools/adb

mkdir /android-sdk/platforms

# Download Android NDK r19c, move /temp_ndk/android-ndk-<version> to /android_ndk
wget -qO- -O temp.zip https://dl.google.com/android/repository/android-ndk-r20-linux-x86_64.zip && unzip -oq temp.zip -d /temp-ndk && mv /temp-ndk/* /android-ndk && rm temp.zip && rm -rf /temp-ndk && ls /android-ndk

apt-get -y remove libprotobuf-dev protobuf-compiler
