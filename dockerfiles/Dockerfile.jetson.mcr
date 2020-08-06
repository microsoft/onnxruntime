# --------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------
# Dockerfile to run ONNXRuntime with CUDA, CUDNN integration

############################################################
## Change to the latest base image for the Jetpack from NGC
## Jetpack 4.4: l4t-base:r32.4.3
############################################################
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-base:r32.4.3
ARG ONNXRUNTIME_REPO=https://github.com/Microsoft/onnxruntime
ARG ONNXRUNTIME_BRANCH=master

FROM ${BASE_IMAGE} as onnxruntime

# Make it so that docker doesn't request user interaction and break
ENV DEBIAN_FRONTEND=noninteractive

# Update repositories and install numpy, opencv, pip, and some other libaries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    libopenblas-dev \
    cmake\
    python3 \
    python3-pip \
    python3-dev \
    python3-opencv \
    libcurl4-openssl-dev \
    libboost-python1.65-dev \
    libpython3-dev \
    libatlas-base-dev \
    libffi-dev \
    libssl-dev \
    g++-arm-linux-gnueabihf \
    git-all && \
    rm -rf /var/lib/apt/lists/* 

RUN pip3 install --upgrade pip
RUN pip3 install setuptools
RUN pip3 install --upgrade numpy

############################################################
# Download ONNX Runtime python wheel published by NVIDIA
# ONNX Runtime v.1.4.0 (for JetPack 4.4 DP): https://nvidia.box.com/shared/static/8sc6j25orjcpl6vhq3a4ir8v219fglng.whl (onnxruntime_gpu-1.4.0-cp36-cp36m-linux_aarch64.whl)
############################################################
WORKDIR /code
RUN wget https://nvidia.box.com/shared/static/8sc6j25orjcpl6vhq3a4ir8v219fglng.whl -O onnxruntime_gpu-1.4.0-cp36-cp36m-linux_aarch64.whl && \
    pip3 install onnxruntime_gpu-1.4.0-cp36-cp36m-linux_aarch64.whl

# Set paths for gpu libraries
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu-override/tegra/:/usr/lib/aarch64-linux-gnu-override/:/usr/lib/aarch64-linux-gnu/:/usr/local/cuda-10.2/targets/aarch64-linux/lib/:/usr/local/cuda-10.2/:/usr/local/cuda/:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/cuda/bin:/usr/bin:/bin:${PATH}


# Setup the license files in the base image from master repo
WORKDIR /code
RUN git clone --single-branch --branch master --recursive https://github.com/Microsoft/onnxruntime onnxruntime &&\
    cp onnxruntime/docs/Privacy.md /code/Privacy.md &&\
    cp onnxruntime/ThirdPartyNotices.txt /code/ThirdPartyNotices.txt

# Cleanup ONNX wheels and repo
WORKDIR /code
RUN rm -rf *.whl && \
    rm -rf onnxruntime

LABEL maintainer="onnxcoredev@microsoft.com"
LABEL description="This is a preview release for the ONNX Runtime on nVidia Jetpack 4.4."
LABEL version="v1.4"
