ARG BASE_IMAGE=mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.0-cudnn7-ubuntu16.04
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        zip \
        curl \
        libcurl4-openssl-dev \
        libssl-dev

RUN conda install -y python=3.7 numpy && \
    conda clean -aqy && \
    rm -rf /opt/miniconda/pkgs && \
    # Dependencies: cmake
    wget --quiet https://github.com/Kitware/CMake/releases/download/v3.14.3/cmake-3.14.3-Linux-x86_64.tar.gz && \
    tar zxf cmake-3.14.3-Linux-x86_64.tar.gz -C /opt && \
    rm cmake-3.14.3-Linux-x86_64.tar.gz

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/cmake-3.14.3-Linux-x86_64/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/lib:${LD_LIBRARY_PATH}
ENV CUDA_HOME /usr/local/cuda
ENV CUDNN_HOME /usr/local/cuda

WORKDIR /onnxruntime

ENTRYPOINT ["./build.sh"]
