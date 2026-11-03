# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used by Zip-Nuget Packaging NoContribOps Pipeline,Zip-Nuget-Java Packaging Pipeline
ARG BASEIMAGE=onnxruntimebuildcache.azurecr.io/internal/azureml/onnxruntime/build/cpu_x64_almalinux8_gcc14_dotnet@sha256:0f8e29b82f43e7092ef4ff70c5ea70fb7c3ebb376816640fe2c177b966323982
FROM $BASEIMAGE

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN dnf install -y perl-core && dnf clean all

ADD scripts /tmp/scripts
RUN --mount=type=secret,id=PIP_INDEX_URL,required=false \
    if [ -f /run/secrets/PIP_INDEX_URL ]; then export PIP_INDEX_URL=$(cat /run/secrets/PIP_INDEX_URL); fi && \
    python3 -m pip install flatbuffers

ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER
