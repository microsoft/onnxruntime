ARG OS_VERSION=16.04
FROM ubuntu:${OS_VERSION}

ARG PYTHON_VERSION=3.5
ARG OPENVINO_VERSION=2018_R5

ADD scripts /tmp/scripts
ENV PATH="/opt/cmake/bin:${PATH}"
RUN /tmp/scripts/install_ubuntu.sh -p ${PYTHON_VERSION} && \
    /tmp/scripts/install_deps.sh

RUN /tmp/scripts/install_openvino.sh -o ${OPENVINO_VERSION} && \
    rm -rf /tmp/scripts

WORKDIR /root

ENV INTEL_CVSDK_DIR /data/dldt

ENV LD_LIBRARY_PATH $INTEL_CVSDK_DIR/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64:$INTEL_CVSDK_DIR/deployment_tools/inference_engine/temp/omp/lib:/usr/local/openblas/lib:$LD_LIBRARY_PATH

ENV PATH $INTEL_CVSDK_DIR/deployment_tools/model_optimizer:$PATH
ENV PYTHONPATH $INTEL_CVSDK_DIR/deployment_tools/model_optimizer:$INTEL_CVSDK_DIR/tools:$PYTHONPATH

ARG BUILD_UID=1000
ARG BUILD_USER=onnxruntimedev
WORKDIR /home/$BUILD_USER
RUN adduser --gecos 'onnxruntime Build User' --disabled-password $BUILD_USER --uid $BUILD_UID
USER $BUILD_USER

