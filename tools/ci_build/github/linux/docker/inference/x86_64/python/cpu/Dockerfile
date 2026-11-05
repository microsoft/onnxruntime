ARG BASEIMAGE=onnxruntimebuildcache.azurecr.io/internal/azureml/onnxruntime/build/cpu_x64_almalinux8_gcc14@sha256:f227af058cce5f442b342c3826bf6ff525452349978d66055729ab140d13fc34
FROM $BASEIMAGE

ADD scripts /tmp/scripts
RUN cd /tmp/scripts && /tmp/scripts/install_centos.sh && rm -rf /tmp/scripts

ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER
