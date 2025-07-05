#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: MIT
#
FROM ubuntu:22.04

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
    git \
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
    && apt-get install -y --no-install-recommends aria2 aria2 && aria2c -q -d /tmp -o cmake-4.0.2-linux-x86_64.tar.gz https://github.com/Kitware/CMake/releases/download/v4.0.2/cmake-4.0.2-linux-x86_64.tar.gz && tar -zxf /tmp/cmake-4.0.2-linux-x86_64.tar.gz --strip=1 -C /usr \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN useradd -m builder
USER builder
