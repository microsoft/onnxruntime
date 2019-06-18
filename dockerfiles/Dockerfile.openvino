#-------------------------------------------------------------------------
# Copyright(C) 2019 Intel Corporation.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

ARG OS_VERSION=16.04
FROM ubuntu:${OS_VERSION}

ARG PYTHON_VERSION=3.5
ARG OPENVINO_VERSION=2018_R5
ARG TARGET_DEVICE=CPU_FP32

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y sudo git bash

ENV PATH="/opt/cmake/bin:${PATH}"
RUN git clone --branch preview-v0.7 --recursive https://github.com/intel/onnxruntime onnxruntime
RUN /onnxruntime/tools/ci_build/github/linux/docker/scripts/install_ubuntu.sh -p ${PYTHON_VERSION} && \
    /onnxruntime/tools/ci_build/github/linux/docker/scripts/install_deps.sh

RUN /onnxruntime/tools/ci_build/github/linux/docker/scripts/install_openvino.sh -o ${OPENVINO_VERSION}

WORKDIR /

ENV INTEL_CVSDK_DIR /data/dldt

ENV LD_LIBRARY_PATH $INTEL_CVSDK_DIR/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64:$INTEL_CVSDK_DIR/deployment_tools/inference_engine/temp/omp/lib:/usr/local/openblas/lib:$LD_LIBRARY_PATH

ENV PATH $INTEL_CVSDK_DIR/deployment_tools/model_optimizer:$PATH
ENV PYTHONPATH $INTEL_CVSDK_DIR/deployment_tools/model_optimizer:$INTEL_CVSDK_DIR/tools:$PYTHONPATH

RUN mkdir -p /onnxruntime/build && \
    python3 /onnxruntime/tools/ci_build/build.py --build_dir /onnxruntime/build --config Release --build_shared_lib --skip_submodule_sync --build_wheel --parallel --use_openvino ${TARGET_DEVICE} && \
    pip3 install /onnxruntime/build/Release/dist/onnxruntime-*linux_x86_64.whl && \
    rm -rf /onnxruntime
