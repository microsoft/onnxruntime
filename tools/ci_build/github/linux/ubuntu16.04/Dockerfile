FROM ubuntu:16.04

ADD install.sh /tmp/install.sh
ENV PATH="/opt/cmake/bin:${PATH}"
RUN /tmp/install.sh && rm /tmp/install.sh

WORKDIR /root

ENV LD_LIBRARY_PATH /usr/local/openblas/lib:$LD_LIBRARY_PATH

ARG BUILD_UID=1000
ARG BUILD_USER=onnxruntimedev
RUN adduser --gecos 'ONNXRuntime Build User' --disabled-password $BUILD_USER --uid $BUILD_UID
USER $BUILD_USER
WORKDIR /home/$BUILD_USER
