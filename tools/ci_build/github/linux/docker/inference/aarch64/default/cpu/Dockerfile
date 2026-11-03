# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used by Zip-Nuget Packaging NoContribOps Pipeline,Zip-Nuget-Java Packaging Pipeline
ARG BASEIMAGE=onnxruntimebuildcache.azurecr.io/internal/azureml/onnxruntime/build/cpu_aarch64_almalinux8_gcc14_dotnet@sha256:5c24c8da952587b1429fc9e8f43fa365fc5f8c8090efeca444b25ae6347fa159
FROM $BASEIMAGE

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN dnf install -y perl-core && dnf clean all

RUN --mount=type=secret,id=PIP_INDEX_URL,required=false \
	if [ -f /run/secrets/PIP_INDEX_URL ]; then export PIP_INDEX_URL=$(cat /run/secrets/PIP_INDEX_URL); fi && \
	python3 -m pip install flatbuffers

ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER
