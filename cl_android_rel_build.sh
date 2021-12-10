#!/bin/bash

set -ev

export ANDROID_HOME=`realpath ~/android_sdk/`
export ANDROID_NDK_HOME=`realpath ~/android_sdk/ndk/23.0.7599858`
export CONFIG=MinSizeRel

./build.sh \
    --cmake_generator Ninja \
    --config ${CONFIG} \
    --android \
    --android_abi arm64-v8a \
    --android_api 29 \
    --use_opencl \
    --use_nnapi \
    --build_java \
    --nnapi_min_api 29 \
    --cmake_extra_defines CMAKE_EXPORT_COMPILE_COMMANDS=ON \
    --skip_submodule_sync --skip_tests \

export STRIP=${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip
export BUILD_DIR=build/Android

# ${STRIP} -S ${BUILD_DIR}/${CONFIG}/onnxruntime_perf_test
# find ${BUILD_DIR}/${CONFIG} -name '*.so' -exec ${STRIP} -S {} \;
# find ${BUILD_DIR}/${CONFIG} -name '*.a' -exec ${STRIP} -S {} \;
