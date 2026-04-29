ARG BASEIMAGE=onnxruntimebuildcache.azurecr.io/internal/azureml/onnxruntime/build/cpu_x64_almalinux8_gcc14:20251017.1

# ---------------------------------------------------------------------------
# Builder stage: build SwiftShader (Google's software Vulkan ICD) from source.
#
# Why SwiftShader instead of Mesa lavapipe?
#   The AlmaLinux 8 base ships an old Mesa lavapipe that rejects Dawn's
#   requested Vulkan apiVersion with VK_ERROR_INCOMPATIBLE_DRIVER. SwiftShader
#   is maintained alongside Dawn for headless CI and is self-contained (one
#   .so + ICD JSON, no Mesa/DRM dependency).
#
# SwiftShader has no release tags, so pin to a commit SHA. The ICD must
# advertise at least the Vulkan apiVersion Dawn requests; picking a SHA from
# Dawn's DEPS is a convenient way to get one known to satisfy that.
#
# This SHA is lifted from Dawn's DEPS (third_party/swiftshader entry) at the
# Dawn commit pinned by ORT in cmake/deps.txt. To refresh on a Dawn bump:
#   https://dawn.googlesource.com/dawn/+/<dawn-commit>/DEPS
# ---------------------------------------------------------------------------
FROM $BASEIMAGE AS swiftshader_builder

ARG SWIFTSHADER_COMMIT=b7b7fd22e5f28079b92412f47f6da4df43e4cd37

RUN dnf install -y git ninja-build && dnf clean all

RUN git -c advice.detachedHead=false init /tmp/swiftshader \
 && cd /tmp/swiftshader \
 && git remote add origin https://swiftshader.googlesource.com/SwiftShader \
 && git fetch --depth 1 origin "${SWIFTSHADER_COMMIT}" \
 && git checkout FETCH_HEAD \
 && git submodule update --init --recursive --depth 1

RUN cmake -S /tmp/swiftshader -B /tmp/swiftshader/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DSWIFTSHADER_BUILD_TESTS=OFF \
      -DSWIFTSHADER_BUILD_PVR=OFF \
      -DSWIFTSHADER_WARNINGS_AS_ERRORS=OFF \
 && cmake --build /tmp/swiftshader/build --target vk_swiftshader

# Stage the artifacts + rewrite the ICD JSON's library_path to an absolute
# path so the Vulkan loader can find the .so from any working directory.
RUN mkdir -p /opt/swiftshader \
 && cp /tmp/swiftshader/build/Linux/libvk_swiftshader.so /opt/swiftshader/ \
 && python3 <<'EOF'
import json
src = '/tmp/swiftshader/build/Linux/vk_swiftshader_icd.json'
dst = '/opt/swiftshader/vk_swiftshader_icd.json'
with open(src) as f:
    icd = json.load(f)
icd['ICD']['library_path'] = '/opt/swiftshader/libvk_swiftshader.so'
with open(dst, 'w') as f:
    json.dump(icd, f, indent=2)
EOF

# ---------------------------------------------------------------------------
# Runtime stage: final test image.
# ---------------------------------------------------------------------------
FROM $BASEIMAGE

ADD scripts /tmp/scripts
RUN cd /tmp/scripts && /tmp/scripts/install_centos.sh && rm -rf /tmp/scripts

# Vulkan loader. The SwiftShader ICD is copied from the builder stage
# below. Callers opt into SwiftShader at `docker run` time via
# VK_ICD_FILENAMES / VK_DRIVER_FILES (see plugin-linux-webgpu-test-stage.yml).
RUN dnf install -y vulkan-loader && dnf clean all

COPY --from=swiftshader_builder /opt/swiftshader /opt/swiftshader

ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER
