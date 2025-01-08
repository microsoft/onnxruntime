# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used by Zip-Nuget Packaging NoContribOps Pipeline,Zip-Nuget-Java Packaging Pipeline
FROM onnxruntimebuildcache.azurecr.io/internal/azureml/onnxruntime/build/cuda12_x64_ubi8_gcc12_dotnet:20241120.3
ARG TRT_VERSION

#Install TensorRT only if TRT_VERSION is not empty
RUN if [ -n "$TRT_VERSION" ]; then  \
    echo "TRT_VERSION is $TRT_VERSION" && \
    dnf -y install  \
    libnvinfer10-${TRT_VERSION}  \
    libnvinfer-headers-devel-${TRT_VERSION}  \
    libnvinfer-devel-${TRT_VERSION}  \
    libnvinfer-lean10-${TRT_VERSION}  \
    libnvonnxparsers10-${TRT_VERSION}  \
    libnvonnxparsers-devel-${TRT_VERSION}  \
    libnvinfer-dispatch10-${TRT_VERSION}  \
    libnvinfer-plugin10-${TRT_VERSION}  \
    libnvinfer-vc-plugin10-${TRT_VERSION}  \
    libnvinfer-bin-${TRT_VERSION}  \
    libnvinfer-plugin10-${TRT_VERSION}  \
    libnvinfer-plugin-devel-${TRT_VERSION}  \
    libnvinfer-vc-plugin-devel-${TRT_VERSION}  \
    libnvinfer-lean-devel-${TRT_VERSION}  \
    libnvinfer-dispatch-devel-${TRT_VERSION}  \
    libnvinfer-headers-plugin-devel-${TRT_VERSION} && \
    dnf clean dbcache ; \
else \
    echo "TRT_VERSION is none skipping Tensor RT Installation" ; \
fi



ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

ENV CUDAHOSTCXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
ADD scripts /tmp/scripts
RUN sed -i 's/enabled\s*=\s*1/enabled = 1\nexclude=dotnet* aspnet* netstandard*/g' /etc/yum.repos.d/ubi.repo && \
	cd /tmp/scripts && /tmp/scripts/install_deps.sh && rm -rf /tmp/scripts
ENV PATH=/usr/lib/jvm/msopenjdk-17/bin:$PATH
ENV JAVA_HOME=/usr/lib/jvm/msopenjdk-17
ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER
