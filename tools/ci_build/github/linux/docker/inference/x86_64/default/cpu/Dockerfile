# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used by Zip-Nuget Packaging NoContribOps Pipeline,Zip-Nuget-Java Packaging Pipeline
FROM onnxruntimebuildcache.azurecr.io/internal/azureml/onnxruntime/build/cpu_x64_almalinux8_gcc14_dotnet:20250724.1

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN python3 -m pip install flatbuffers

ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER
