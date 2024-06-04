# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used by Zip-Nuget Packaging NoContribOps Pipeline,Zip-Nuget-Java Packaging Pipeline
FROM onnxruntimebuildcache.azurecr.io/internal/azureml/onnxruntime/build/cuda12_2_x64_ubi8_gcc12_dotnet:20240531.1
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

# install cudnn 9 for cuda 12
RUN if [ $(echo $CUDA_VERSION | cut -d"." -f1) -ge 12 ]; then \
    wget https://developer.download.nvidia.com/compute/cudnn/9.1.1/local_installers/cudnn-local-repo-rhel9-9.1.1-1.0-1.x86_64.rpm ; \
    dnf -y install cudnn-local-repo-rhel9-9.1.1-1.0-1.x86_64.rpm ; \
    dnf clean all ; \
    sudo dnf -y install cudnn ; \
fi