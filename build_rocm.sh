#!/bin/bash

usage() { echo "Usage $0 [--no-config]" 1>&2; exit 1; }

config_cmake=true

while getopts ":h-:" optchar; do
    case "$optchar" in
        -)  case "$OPTARG" in
                no-config) config_cmake=false ;;
            esac;;
        *) echo "config"; usage ;;
    esac
done

THIS_DIR=$(dirname $(realpath $0))

set -ex

build_dir_suffix=""
build_dir="build_rocm${build_dir_suffix}"
config="Release"

rocm_home="/opt/rocm"

if $config_cmake; then
    rm -f  ${THIS_DIR}/${build_dir}/${config}/*.so
    # rm -fr ${THIS_DIR}/${build_dir}/${config}/build/lib

    ${THIS_DIR}/build.sh \
        --build_dir ${THIS_DIR}/${build_dir} \
        --config ${config} \
        --cmake_generator Ninja \
        --cmake_extra_defines \
            CMAKE_HIP_ARCHITECTURES=gfx908,gfx90a \
            CMAKE_EXPORT_COMPILE_COMMANDS=ON \
            onnxruntime_BUILD_KERNEL_EXPLORER=ON \
            onnxruntime_USE_COMPOSABLE_KERNEL=ON \
        --use_cache \
        --use_rocm \
        --rocm_home=/opt/rocm-5.4.3 --nccl_home=/opt/rocm \
        --rocm_version 5.4.3 \
        --enable_rocm_profiling \
        --enable_training \
        --build_wheel \
        --skip_submodule_sync --skip_tests \

        # --use_migraphx \

    # ${THIS_DIR}/create-fake-dist-info.sh ${THIS_DIR}/${build_dir}/${config}/build/lib/
else
    printf "\n\tSkipping config\n\n"
    cmake --build ${THIS_DIR}/${build_dir}/${config}
fi
