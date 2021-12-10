#!/bin/bash

set -ev

export ANDROID_HOME=`realpath ~/android_sdk/`
export ANDROID_NDK_HOME=`realpath ~/android_sdk/ndk/23.0.7599858`

./build.sh \
    --cmake_generator Ninja \
    --config Debug \
    --android \
    --use_opencl \
    --cmake_extra_defines CMAKE_EXPORT_COMPILE_COMMANDS=ON \
    --skip_submodule_sync --skip_tests \
