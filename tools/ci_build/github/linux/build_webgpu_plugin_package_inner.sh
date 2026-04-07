#!/bin/bash
# Inner script that runs inside the Docker container for the WebGPU plugin build.
# Separated from the outer script to avoid shell-in-Docker quoting issues.

BUILD_CONFIG="${1:-Release}"
EXTRA_CMAKE_DEFINES="${2:-}"

# === [DIAG] Pre-build diagnostics ===
echo "=== [DIAG] Pre-build: system Python environment ==="
python3 /onnxruntime_src/tools/ci_build/github/linux/diag_jinja2_env.py || true

# Run the actual build (no set -e so post-build diagnostics still run)
/usr/bin/python3 /onnxruntime_src/tools/ci_build/build.py \
    --build_dir /build \
    --config "${BUILD_CONFIG}" \
    --skip_submodule_sync \
    --parallel \
    --use_binskim_compliant_compile_flags \
    --use_webgpu shared_lib \
    --wgsl_template static \
    --disable_rtti \
    --enable_lto \
    --enable_onnx_tests \
    --use_vcpkg \
    --use_vcpkg_ms_internal_asset_cache \
    --update \
    --build \
    --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=ON ${EXTRA_CMAKE_DEFINES}
BUILD_EXIT=$?

# === [DIAG] Post-build diagnostics (runs even if build failed) ===
echo "=== [DIAG] Post-build: inspect Dawn-fetched jinja2/markupsafe ==="
python3 /onnxruntime_src/tools/ci_build/github/linux/diag_jinja2_env.py /build || true

exit $BUILD_EXIT
