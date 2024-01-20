#!/bin/bash

set -ex

cd $(dirname $0)

pip uninstall -y onnxruntime-training

CONFIG=Release
BUILD_DIR=build/Linux/$CONFIG

rm -rf $BUILD_DIR/build/*
rm -rf $BUILD_DIR/*.so
rm -f $BUILD_DIR/dist/*.whl

bash ./build.sh \
    --config Release --enable_training --build_wheel --update --build --cmake_generator Ninja --skip_tests \
    --use_rocm --rocm_version=${ROCM_VERSION} --rocm_home /opt/rocm --enable_nccl \
    --nccl_home /opt/rocm --allow_running_as_root \
    --parallel --skip_tests --skip_submodule_sync \
    --enable_rocm_profiling \
    --cmake_extra_defines onnxruntime_USE_COMPOSABLE_KERNEL=ON \
                          onnxruntime_USE_ROCBLAS_EXTENSION_API=ON \
                          onnxruntime_USE_HIPBLASLT=ON \
                          onnxruntime_USE_TRITON_KERNEL=OFF \
                          onnxruntime_BUILD_KERNEL_EXPLORER=ON \
                          CMAKE_HIP_ARCHITECTURES=gfx90a && \
pip install build/Linux/Release/dist/*.whl
