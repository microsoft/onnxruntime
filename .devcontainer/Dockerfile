# --------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------
# Dockerfile to Develop inside Dev Container

ARG VARIANT=ubuntu-20.04
FROM mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}

ENV DEBIAN_FRONTEND=noninteractive

# Install common packages
RUN apt-get update \
    && apt-get -y install build-essential cppcheck valgrind clang lldb llvm gdb \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install additional OS packages.
RUN apt-get update \
    && apt-get -y install --no-install-recommends \
    python3-dev \
    python3 \
    ca-certificates \
    python3-numpy \
    python3-setuptools \
    python3-wheel \
    python3-pip \
    g++ \
    gcc \
    ninja-build \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install newer CMake version
RUN apt-get update \
    && apt-get install -y --no-install-recommends aria2 aria2 && aria2c -q -d /tmp -o cmake-3.27.3-linux-x86_64.tar.gz https://github.com/Kitware/CMake/releases/download/v3.27.3/cmake-3.27.3-linux-x86_64.tar.gz && tar -zxf /tmp/cmake-3.27.3-linux-x86_64.tar.gz --strip=1 -C /usr \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*
