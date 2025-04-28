# Use the specified UBI8 base image with GCC 14
ARG BASEIMAGE="onnxruntimebuildcache.azurecr.io/internal/azureml/onnxruntime/build/cpu_x64_ubi8_gcc14:20250124.1"
FROM ${BASEIMAGE}

ARG BUILD_UID=1000
ARG BUILD_USER=onnxruntimedev

# Switch to root for system-level installations
USER root
WORKDIR /

RUN dnf install -y --nodocs \
        wget \
        tar \
        gzip \
        git \
        numactl-libs \
        ocl-icd \
    && dnf clean all \
    && rm -rf /var/cache/dnf

ENV INTEL_OPENVINO_DIR=/opt/intel/openvino_2025.1.0
ARG OPENVINO_PACKAGE_URL=https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.1/linux/openvino_toolkit_rhel8_2025.1.0.18503.6fec06580ab_x86_64.tgz
ARG TEMP_DIR=/tmp/openvino_installer

RUN mkdir -p ${TEMP_DIR} && \
    cd ${TEMP_DIR} && \
    wget -O openvino_toolkit.tgz ${OPENVINO_PACKAGE_URL} && \
    tar -xzf openvino_toolkit.tgz && \
    OV_FOLDER_NAME=$(ls -d openvino_toolkit_rhel8_*) && \
    mkdir -p /opt/intel && \
    mv ${OV_FOLDER_NAME} ${INTEL_OPENVINO_DIR} && \
    ln -s ${INTEL_OPENVINO_DIR} /opt/intel/openvino && \
    # Clean up installer files
    cd / && rm -rf ${TEMP_DIR}

RUN dnf install -y --nodocs \
        libedit \
        ocl-icd \
        https://repositories.intel.com/graphics/rhel/8.6/intel-gmmlib-22.3.1-i529.el8.x86_64.rpm \
        https://repositories.intel.com/graphics/rhel/8.6/intel-igc-core-1.0.12504.6-i537.el8.x86_64.rpm \
        https://repositories.intel.com/graphics/rhel/8.6/intel-igc-opencl-1.0.12504.6-i537.el8.x86_64.rpm \
        https://repositories.intel.com/graphics/rhel/8.6/intel-opencl-22.43.24595.35-i538.el8.x86_64.rpm \
        https://repositories.intel.com/graphics/rhel/8.6/intel-level-zero-gpu-1.3.24595.35-i538.el8.x86_64.rpm \
        https://repositories.intel.com/graphics/rhel/8.6/level-zero-1.8.8-i524.el8.x86_64.rpm \
    && dnf clean all \
    && rm -rf /var/cache/dnf

# --- Environment Setup ---
ENV LD_LIBRARY_PATH=${INTEL_OPENVINO_DIR}/runtime/3rdparty/hddl/lib:${INTEL_OPENVINO_DIR}/runtime/3rdparty/tbb/lib:${INTEL_OPENVINO_DIR}/runtime/lib/intel64:${INTEL_OPENVINO_DIR}/tools/compile_tool:${INTEL_OPENVINO_DIR}/extras/opencv/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV OpenCV_DIR=${INTEL_OPENVINO_DIR}/extras/opencv/cmake
ENV PATH=${INTEL_OPENVINO_DIR}/runtime/bin/intel64:${PATH}
ENV TBB_DIR=${INTEL_OPENVINO_DIR}/runtime/3rdparty/tbb/cmake
ENV ngraph_DIR=${INTEL_OPENVINO_DIR}/runtime/cmake
ENV OpenVINO_DIR=${INTEL_OPENVINO_DIR}/runtime/cmake
ENV PKG_CONFIG_PATH=${INTEL_OPENVINO_DIR}/runtime/lib/intel64/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}
ENV OpenVINORootDir=${INTEL_OPENVINO_DIR}

# --- Build User Setup ---
# Create the user with the specified UID and add to the 'video' group for GPU access.
# Also add to 'users' group which is common practice.
RUN groupmod -g 44 video || groupadd -g 44 video && \
    useradd --shell /bin/bash --uid ${BUILD_UID} --gid users --groups video --create-home --home-dir /home/${BUILD_USER} ${BUILD_USER} && \
    chown ${BUILD_USER}:users /home/${BUILD_USER}

# Switch to the build user
USER ${BUILD_USER}
WORKDIR /home/${BUILD_USER}
