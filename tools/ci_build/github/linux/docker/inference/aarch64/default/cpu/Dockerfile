# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used by Zip-Nuget Packaging NoContribOps Pipeline,Zip-Nuget-Java Packaging Pipeline
ARG BASEIMAGE=onnxruntimebuildcache.azurecr.io/internal/azureml/onnxruntime/build/cpu_aarch64_almalinux8_gcc14_dotnet:20251008.2
FROM $BASEIMAGE

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN python3 -m pip install flatbuffers

# Install azcopy for vcpkg binary caching via Azure Blob Storage
RUN curl -sSL https://aka.ms/downloadazcopy-v10-linux-arm64 -o /tmp/azcopy.tar.gz \
    && tar -xf /tmp/azcopy.tar.gz --strip-components=1 -C /usr/local/bin --wildcards '*/azcopy' \
    && chmod +x /usr/local/bin/azcopy \
    && rm /tmp/azcopy.tar.gz

ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER
