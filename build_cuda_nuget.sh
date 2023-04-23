#!/bin/bash

set -ex

THIS_DIR=$(dirname $(realpath $0))

build_dir="build_cuda"
config="Release"

${THIS_DIR}/build.sh \
    --build_dir ${THIS_DIR}/${build_dir} \
    --config ${config} \
    --cmake_generator Ninja \
    --cmake_extra_defines \
        CMAKE_CUDA_ARCHITECTURES=70 \
        CMAKE_EXPORT_COMPILE_COMMANDS=ON \
    --use_cuda \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr \
    --skip_submodule_sync --skip_tests \
    --build_nuget \
    --msbuild_extra_options \
        /p:SelectedTargets=Net6 /p:Net6Targets=net6.0 /p:TargetFrameworks=netstandard2.0 /p:IsLinuxBuild=true
