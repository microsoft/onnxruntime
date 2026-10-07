ARG BASEIMAGE=onnxruntimebuildcache.azurecr.io/internal/azureml/onnxruntime/build/cpu_aarch64_almalinux8_gcc14:20251017.1
FROM $BASEIMAGE

ADD scripts /tmp/scripts
RUN --mount=type=secret,id=PIP_INDEX_URL,required=false \
    --mount=type=secret,id=CARGO_CONFIG,required=false \
    --mount=type=secret,id=SYSTEM_ACCESSTOKEN,required=false \
    if [ -f /run/secrets/PIP_INDEX_URL ]; then export PIP_INDEX_URL=$(cat /run/secrets/PIP_INDEX_URL); fi && \
    if [ -f /run/secrets/CARGO_CONFIG ]; then mkdir -p /.cargo && cp /run/secrets/CARGO_CONFIG /.cargo/config.toml; fi && \
    if [ -f /run/secrets/SYSTEM_ACCESSTOKEN ]; then export CARGO_REGISTRIES_ONNXRUNTIME_TOKEN="Basic $(printf ':%s' "$(cat /run/secrets/SYSTEM_ACCESSTOKEN)" | base64 -w0)"; fi && \
    cd /tmp/scripts && /tmp/scripts/install_centos.sh && /tmp/scripts/install_deps.sh && rm -rf /tmp/scripts

ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER
