FROM debian:buster

RUN apt-get update && apt-get install -y apt-utils && apt-get install -y gcc-8-aarch64-linux-gnu g++-8-aarch64-linux-gnu aria2 make unzip git \
    && aria2c -q -d /tmp https://github.com/Kitware/CMake/releases/download/v3.21.2/cmake-3.21.2-linux-x86_64.tar.gz \
    && aria2c -q -d /tmp https://github.com/protocolbuffers/protobuf/releases/download/v3.16.0/protoc-3.16.0-linux-x86_64.zip \
    && tar -zxf /tmp/cmake-3.21.2-linux-x86_64.tar.gz --strip=1 -C /usr \
    && unzip -d /usr /tmp/protoc-3.16.0-linux-x86_64.zip
    
