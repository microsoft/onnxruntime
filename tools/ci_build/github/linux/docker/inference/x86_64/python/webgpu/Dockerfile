ARG BASEIMAGE=onnxruntimebuildcache.azurecr.io/internal/azureml/onnxruntime/build/cpu_x64_almalinux8_gcc14:20251017.1
FROM $BASEIMAGE

ADD scripts /tmp/scripts
RUN cd /tmp/scripts && /tmp/scripts/install_centos.sh && rm -rf /tmp/scripts

# Install the Vulkan loader plus Mesa's lavapipe ICD (a software/CPU
# Vulkan implementation based on LLVMpipe). This lets Dawn/WebGPU run on
# CI agents that do not expose a GPU to the container. Callers that want
# to pin Vulkan to lavapipe should set VK_ICD_FILENAMES / VK_DRIVER_FILES
# at `docker run` time (see plugin-linux-webgpu-stage.yml).
#
# vulkan-tools is included for diagnostic purposes (vulkaninfo). It can
# be removed once the WebGPU EP test is stable.
RUN dnf install -y vulkan-loader mesa-vulkan-drivers vulkan-tools && dnf clean all

ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER
