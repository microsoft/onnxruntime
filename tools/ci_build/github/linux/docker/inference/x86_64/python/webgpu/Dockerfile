ARG BASEIMAGE=onnxruntimebuildcache.azurecr.io/internal/azureml/onnxruntime/build/cpu_x64_almalinux8_gcc14:20251017.1
FROM $BASEIMAGE

ADD scripts /tmp/scripts
RUN cd /tmp/scripts && /tmp/scripts/install_centos.sh && rm -rf /tmp/scripts

# Install Vulkan loader so Dawn/WebGPU can access the GPU at runtime.
# The NVIDIA Container Toolkit injects the Vulkan ICD driver when the
# container is run with --gpus all.
# vulkan-tools is included for diagnostic purposes (vulkaninfo).
# TODO: remove vulkan-tools once the WebGPU EP test is stable in CI.
RUN dnf install -y vulkan-loader vulkan-tools && dnf clean all

# Tell the NVIDIA Container Toolkit to inject the graphics/Vulkan driver
# in addition to the default compute/utility. Without "graphics", the
# NVIDIA Vulkan ICD and userspace driver are not made available inside
# the container, which produces:
#   setup_loader_term_phys_devs: Failed to detect any valid GPUs in the current config
# when Dawn/WebGPU tries to enumerate Vulkan physical devices.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV NVIDIA_VISIBLE_DEVICES=all

ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER
