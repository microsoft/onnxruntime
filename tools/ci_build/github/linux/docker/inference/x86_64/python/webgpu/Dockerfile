ARG BASEIMAGE=onnxruntimebuildcache.azurecr.io/internal/azureml/onnxruntime/build/cpu_x64_almalinux8_gcc14:20251017.1
FROM $BASEIMAGE

ADD scripts /tmp/scripts
RUN cd /tmp/scripts && /tmp/scripts/install_centos.sh && rm -rf /tmp/scripts

# Install Vulkan loader so Dawn/WebGPU can access the GPU at runtime.
# The NVIDIA Container Toolkit injects the Vulkan ICD driver when the
# container is run with --gpus all.
RUN dnf install -y vulkan-loader && dnf clean all

ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER
