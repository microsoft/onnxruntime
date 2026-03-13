ARG BASEIMAGE=onnxruntimebuildcache.azurecr.io/internal/azureml/onnxruntime/build/cpu_aarch64_almalinux8_gcc14:20251017.1
FROM $BASEIMAGE

ADD scripts /tmp/scripts
RUN cd /tmp/scripts && /tmp/scripts/install_centos.sh && /tmp/scripts/install_deps.sh && rm -rf /tmp/scripts

# Install azcopy for vcpkg binary caching via Azure Blob Storage
RUN curl -sSL https://aka.ms/downloadazcopy-v10-linux-arm64 -o /tmp/azcopy.tar.gz \
    && tar -xf /tmp/azcopy.tar.gz --strip-components=1 -C /usr/local/bin --wildcards '*/azcopy' \
    && chmod +x /usr/local/bin/azcopy \
    && rm /tmp/azcopy.tar.gz

ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER
