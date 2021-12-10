#!/bin/bash

set -ev

./build.sh \
    --cmake_generator Ninja \
    --config Release \
    --cmake_extra_defines CMAKE_EXPORT_COMPILE_COMMANDS=ON \
    --use_opencl \
    --build_wheel \
    --skip_submodule_sync --skip_tests \
