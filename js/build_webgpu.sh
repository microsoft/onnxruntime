#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# build_webgpu.sh --- build onnxruntime-web with WebGPU EP
#
# Usage:
#   build_webgpu.sh config [clean]
#
# Options:
#   config      Build configuration, "d" (Debug) or "r" (Release)
#   clean       Perform a clean build (optional)

# Determine the root directory of the project (one level up from the script's directory)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build_webgpu"

CONFIG=""
CONFIG_EXTRA_FLAG="" # This will be empty by default

# Parse config argument
if [ "$1" = "d" ]; then
    CONFIG="Debug"
    CONFIG_EXTRA_FLAG="--enable_wasm_profiling --wasm_run_tests_in_browser --cmake_extra_defines onnxruntime_ENABLE_WEBASSEMBLY_OUTPUT_OPTIMIZED_MODEL=1 --enable_wasm_debug_info"
elif [ "$1" = "r" ]; then
    CONFIG="Release"
    CONFIG_EXTRA_FLAG="--enable_wasm_api_exception_catching --disable_rtti"
else
    echo "Error: Invalid configuration \"$1\"."
    echo "Configuration must be 'd' (Debug) or 'r' (Release)."
    echo "Usage: $0 [d|r] [clean]"
    exit 1
fi

CLEAN_BUILD_REQUESTED=false
if [ "$2" = "clean" ]; then
    CLEAN_BUILD_REQUESTED=true
fi

# Perform clean if requested
if [ "$CLEAN_BUILD_REQUESTED" = true ]; then
    echo "--- Performing clean build ---"
    if [ -d "$BUILD_DIR" ]; then
        echo "Removing build directory: $BUILD_DIR"
        rm -rf "$BUILD_DIR"
    fi

    echo "Synchronizing and updating submodules..."
    pushd "$ROOT_DIR" > /dev/null
    git submodule sync --recursive
    git submodule update --init --recursive
    popd > /dev/null
fi

# Determine if npm ci needs to be run
# It needs to run if:
# 1. A clean build was requested (which implies js/web/dist will be missing or stale)
# 2. The js/web/dist directory does not exist (e.g., first build or manually removed)
PERFORM_NPM_CI=false
if [ "$CLEAN_BUILD_REQUESTED" = true ]; then
    PERFORM_NPM_CI=true
elif [ ! -d "$ROOT_DIR/js/web/dist" ]; then
    echo "Directory $ROOT_DIR/js/web/dist not found."
    PERFORM_NPM_CI=true
fi

if [ "$PERFORM_NPM_CI" = true ]; then
    echo "--- Running npm ci and pulling WASM artifacts ---"
    echo "Running npm ci in $ROOT_DIR/js"
    pushd "$ROOT_DIR/js" > /dev/null
    npm ci
    popd > /dev/null

    echo "Running npm ci in $ROOT_DIR/js/common"
    pushd "$ROOT_DIR/js/common" > /dev/null
    npm ci
    popd > /dev/null

    echo "Running npm ci and pull:wasm in $ROOT_DIR/js/web"
    pushd "$ROOT_DIR/js/web" > /dev/null
    npm ci
    npm run pull:wasm
    popd > /dev/null
fi

echo "--- Building WebAssembly modules ---"

echo "Calling $ROOT_DIR/build.sh to build WebAssembly..."
# Note: If $CONFIG_EXTRA_FLAG is empty, it will be omitted from the command due to shell expansion.
"$ROOT_DIR/build.sh" \
    --config "$CONFIG" \
    --parallel \
    ${CONFIG_EXTRA_FLAG} \
    --skip_submodule_sync \
    --build_wasm \
    --target onnxruntime_webassembly \
    --skip_tests \
    --enable_wasm_simd \
    --enable_wasm_threads \
    --use_webnn \
    --use_webgpu \
    --build_dir "$BUILD_DIR"

# The 'set -e' command at the beginning of the script ensures that the script will exit
# immediately if the build.sh command (or any other command) fails.

echo "--- Copying build artifacts ---"
# Ensure the dist directory exists before copying files
mkdir -p "$ROOT_DIR/js/web/dist"

echo "Copying ort-wasm-simd-threaded.asyncify.wasm to $ROOT_DIR/js/web/dist/"
cp -f "$BUILD_DIR/$CONFIG/ort-wasm-simd-threaded.asyncify.wasm" "$ROOT_DIR/js/web/dist/"

echo "Copying ort-wasm-simd-threaded.asyncify.mjs to $ROOT_DIR/js/web/dist/"
cp -f "$BUILD_DIR/$CONFIG/ort-wasm-simd-threaded.asyncify.mjs" "$ROOT_DIR/js/web/dist/"

echo "--- WebGPU build process completed successfully ---"
