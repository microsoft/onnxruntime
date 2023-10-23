FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Define argument variables
ARG ONNXRUNTIME_REPO=https://github.com/microsoft/onnxruntime
# TODO: Change to 'main' branch once all LLaMA-2 changes are merged
ARG ONNXRUNTIME_BRANCH=kvaishnavi/llama_int4_gqa
ARG CMAKE_CUDA_ARCHITECTURES=70;75;80
# ARG TENSORRT_VERSION="8.6.1-6+cuda11.8"

# Define environment info
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV PATH=/usr/src/tensorrt/bin:${PATH}
ENV PATH=/home/cmake-${CMAKE_VERSION}-linux-x86_64/bin:${PATH}
ENV DEBIAN_FRONTEND=noninteractive
ENV LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install packages with apt
RUN apt-get update --fix-missing 
RUN apt install -y \
    sudo \
    vim \
    git \
    bash \
    wget \
    libz-dev \
    unattended-upgrades \
    python3 \
    python3-dev \
    python3-pip \
    python3-wheel \
    cmake \
    gcc \
    g++ \
    make

# Run initial setup commands
RUN unattended-upgrade
RUN cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip
RUN pip install --upgrade pip
RUN pip install "setuptools>=41.0.0"

# # Install TensorRT and compile trtexec
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
# RUN apt-get update
# RUN apt-get install -y \
#     libnvinfer8=${TENSORRT_VERSION} \
#     libnvonnxparsers8=${TENSORRT_VERSION} \
#     libnvparsers8=${TENSORRT_VERSION} \
#     libnvinfer-plugin8=${TENSORRT_VERSION} \
#     libnvinfer-dev=${TENSORRT_VERSION} \
#     libnvonnxparsers-dev=${TENSORRT_VERSION} \
#     libnvparsers-dev=${TENSORRT_VERSION} \
#     libnvinfer-plugin-dev=${TENSORRT_VERSION} \
#     python3-libnvinfer=${TENSORRT_VERSION} \
#     libnvinfer-samples=${TENSORRT_VERSION}
# RUN cd /usr/src/tensorrt/samples/trtexec && make

# Install necessary software with apt/pip/wget before building ORT
WORKDIR /home
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    zip \
    ca-certificates \
    build-essential \
    curl \
    libcurl4-openssl-dev \
    libssl-dev
RUN pip install numpy packaging
RUN pip install "wheel>=0.35.1"

# Prepare ONNX Runtime repository & build ONNX Runtime
RUN git clone --single-branch --branch ${ONNXRUNTIME_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime

# Run the below command in your local dev environment
# RUN cd onnxruntime && \
#     /bin/sh ./build.sh --parallel --build_shared_lib --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --use_tensorrt --tensorrt_home /usr/lib/x86_64-linux-gnu/ --config Release --build_wheel --skip_tests --skip_submodule_sync --cmake_extra_defines '"CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}'"' && \
#     pip install /home/onnxruntime/build/Linux/Release/dist/*.whl && cd ..

# To save time, build the wheel locally with the above instructions (example .whl name listed below) and uncomment the below lines
# ADD onnxruntime_gpu-1.17.0-cp38-cp38-linux_x86_64.whl .
# RUN pip install onnxruntime_gpu-1.17.0-cp38-cp38-linux_x86_64.whl

# When LLaMA-2 changes are all in ORT main (where MM = month, DD = date)
# RUN pip install ort-nightly-gpu==1.17.0.dev2023MMDD001 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/

# Install necessary packages with pip after building ORT
RUN pip install --upgrade pip
RUN pip install --upgrade onnx coloredlogs psutil py3nvml onnxconverter_common numpy sympy

RUN pip install torch torchaudio torchvision --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu118
RUN pip install transformers==4.33.2
RUN pip install git+https://github.com/huggingface/optimum
