# Refer to https://github.com/RadeonOpenCompute/ROCm-docker/blob/master/dev/Dockerfile-ubuntu-22.04-complete
FROM ubuntu:22.04

ARG ROCM_VERSION=6.2
ARG AMDGPU_VERSION=${ROCM_VERSION}
# Default to 1 so that we can share the layers between docker images for ROCm and MIGraphX builds to reduce storage.
ARG USE_MIGRAPHX=1

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV MIGRAPHX_DISABLE_FAST_GELU=1

RUN echo 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' > /etc/apt/preferences.d/rocm-pin-600

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    libnuma-dev \
    gnupg \
    sudo \
    libelf1 \
    kmod \
    file \
    libstdc++6 \
    python3 \
    python3-pip \
    python3.10-dev \
    python3.10-venv \
    build-essential \
    locales \
    git

RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

# Add ROCm repository and install ROCm and optional MIGraphX
RUN curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \
    echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/$ROCM_VERSION/ jammy main" | tee /etc/apt/sources.list.d/rocm.list && \
    echo "deb [arch=amd64] https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/ubuntu jammy main" | tee /etc/apt/sources.list.d/amdgpu.list && \
    migraphx=$( [ "$USE_MIGRAPHX" -eq 1 ] && echo "migraphx" || echo "" ) && \
    apt-get update && apt-get install -y rocm-dev rocm-libs $migraphx && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install CMake
ENV CMAKE_VERSION=3.30.5
RUN wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \
    tar -zxf cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz --strip-components=1 -C /usr && \
    rm cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz

# Install ccache
ENV CCACHE_VERSION=4.10.2
RUN wget -q https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-x86_64.tar.xz && \
    tar -xf ccache-${CCACHE_VERSION}-linux-x86_64.tar.xz && \
    cp ccache-${CCACHE_VERSION}-linux-x86_64/ccache /usr/bin && \
    rm -rf ccache-${CCACHE_VERSION}-linux-x86_64*

# Set up virtual environment for Python and install dependencies
WORKDIR /ort
COPY scripts/requirements.txt /ort/
RUN python3 -m venv /ort/env && . /ort/env/bin/activate && \
    pip install --upgrade pip && \
    pip install -r /ort/requirements.txt && \
    pip install psutil ml_dtypes pytest-xdist pytest-rerunfailures scipy

# Clone and install CuPy with ROCm support
RUN git clone https://github.com/ROCm/cupy.git && cd cupy && \
    git checkout 432a8683351d681e00903640489cb2f4055d2e09 && \
    export CUPY_INSTALL_USE_HIP=1 && \
    export ROCM_HOME=/opt/rocm && \
    export HCC_AMDGPU_TARGET=gfx906,gfx908,gfx90a && \
    git submodule update --init && \
    pip install -e . --no-cache-dir -vvvv && \
    cd .. && rm -rf cupy

CMD ["/bin/bash"]
