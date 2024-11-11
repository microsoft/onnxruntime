# The default ARGs are for cuda 11.8 with cudnn8, TensorRT is optional
# Please overwrite BASEIMAGE, TRT_VERSION and other arguments with
# --docker-build-args ' --build-arg BASEIMAGE=other_base_image --build-arg TRT_VERSION=other_trt_version etc...'
# for other cuda version and TRT version
ARG BASEIMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubi8

FROM $BASEIMAGE
ARG TRT_VERSION=10.6.0.26-1.cuda11.8

#Install TensorRT only if TRT_VERSION is not empty
RUN if [ -n "${TRT_VERSION}" ]; then  \
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
    echo "TRT_VERSION is x${TRT_VERSION} skipping Tensor RT Installation" ; \
fi

ENV PATH=/usr/local/cuda/bin:$PATH
ENV CUDA_MODULE_LOADING="LAZY"

ADD scripts /tmp/scripts
RUN cd /tmp/scripts && /tmp/scripts/install_centos.sh && rm -rf /tmp/scripts

ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER
