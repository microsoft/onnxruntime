FROM onnxruntimebuildcache.azurecr.io/artifact/4cb52e54-39ee-48a0-bdee-8769e66888cb/buddy/internal/azureml/onnxruntime/build/cpu_x64_ubi8_gcc12:20241231.7

ADD scripts /tmp/scripts
RUN cd /tmp/scripts && /tmp/scripts/install_centos.sh && rm -rf /tmp/scripts

ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER

