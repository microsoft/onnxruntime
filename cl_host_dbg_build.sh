#!/bin/bash

set -ev

./build.sh \
    --cmake_generator Ninja \
    --config Debug \
    --cmake_extra_defines CMAKE_EXPORT_COMPILE_COMMANDS=ON onnxruntime_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD=ON\
    --use_opencl \
    --use_nnapi \
    --build_wheel \
    --minimal_build disabled \
    --skip_submodule_sync --skip_tests \

    # --build_java \
