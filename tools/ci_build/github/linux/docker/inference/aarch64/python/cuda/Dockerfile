# The default ARGs are for cuda 12.8 with cudnn9, TensorRT is optional
# Please overwrite BASEIMAGE, TRT_VERSION and other arguments with
# --docker-build-args ' --build-arg BASEIMAGE=other_base_image --build-arg TRT_VERSION=other_trt_version etc...'
# for other cuda version and TRT version
ARG BASEIMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubi8

FROM $BASEIMAGE
ARG TRT_VERSION
# For aarch64 tar-based TensorRT install
ARG TRT_DOWNLOAD_URL=""
ARG TENSORRT_ROOT=/opt/tensorrt

# Install TensorRT: use tar download for aarch64 since RPMs are not available
RUN set -eux; \
    if [ -z "${TRT_VERSION}" ]; then \
        echo "TRT_VERSION is empty; skipping TensorRT installation"; \
    elif [ -n "${TRT_DOWNLOAD_URL}" ]; then \
        echo "Installing TensorRT ${TRT_VERSION} from tar"; \
        mkdir -p /tmp/trt "${TENSORRT_ROOT}"; \
        curl -fsSL "${TRT_DOWNLOAD_URL}" -o /tmp/trt/tensorrt.tar.gz; \
        tar -xzf /tmp/trt/tensorrt.tar.gz -C /tmp/trt; \
        extracted_dir="$(find /tmp/trt -mindepth 1 -maxdepth 1 -type d | head -n 1)"; \
        cp -a "${extracted_dir}/." "${TENSORRT_ROOT}/"; \
        rm -rf /tmp/trt; \
        if [ -d "${TENSORRT_ROOT}/targets/sbsa-linux-gnu/lib" ] && [ ! -e "${TENSORRT_ROOT}/lib" ]; then \
            ln -s "${TENSORRT_ROOT}/targets/sbsa-linux-gnu/lib" "${TENSORRT_ROOT}/lib"; \
        fi; \
        if [ -d "${TENSORRT_ROOT}/targets/sbsa-linux-gnu/include" ] && [ ! -e "${TENSORRT_ROOT}/include" ]; then \
            ln -s "${TENSORRT_ROOT}/targets/sbsa-linux-gnu/include" "${TENSORRT_ROOT}/include"; \
        fi; \
    else \
        echo "TRT_VERSION is ${TRT_VERSION} but no TRT_DOWNLOAD_URL provided; skipping"; \
    fi

ENV TENSORRT_ROOT=${TENSORRT_ROOT}
ENV PATH=${TENSORRT_ROOT}/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=${TENSORRT_ROOT}/lib:${LD_LIBRARY_PATH}
ENV CPATH=${TENSORRT_ROOT}/include:${CPATH}
ENV CUDA_MODULE_LOADING="LAZY"

ADD scripts /tmp/scripts
RUN cd /tmp/scripts && /tmp/scripts/install_centos.sh && rm -rf /tmp/scripts

ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER
