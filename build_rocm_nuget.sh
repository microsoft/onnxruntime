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

build_dir_suffix="_nuget"
build_dir="build_rocm${build_dir_suffix}"
config="Release"

rocm_home="/opt/rocm"

if $config_cmake; then
    rm -f ${THIS_DIR}/${build_dir}/${config}/*.nupkg
    rm -rf ${THIS_DIR}/${build_dir}/${config}/nuget-local-artifacts/
    # rm -f /home/guangyunhan/rocm-dotnet/*.nupkg

    rm -f  ${THIS_DIR}/${build_dir}/${config}/*.so

    ${THIS_DIR}/build.sh \
        --build_dir ${THIS_DIR}/${build_dir} \
        --config ${config} \
        --cmake_generator Ninja \
        --cmake_extra_defines \
            CMAKE_HIP_ARCHITECTURES=gfx908,gfx90a \
            CMAKE_EXPORT_COMPILE_COMMANDS=ON \
            onnxruntime_USE_COMPOSABLE_KERNEL=OFF \
        --use_rocm \
        --parallel=40 \
        --rocm_home /opt/rocm --nccl_home=/opt/rocm \
        --skip_submodule_sync --skip_tests \
        --build_nuget \
        --msbuild_extra_options \
            /p:SelectedTargets=Net6 /p:Net6Targets=net6.0 /p:TargetFrameworks=netstandard2.0 /p:IsLinuxBuild=true

        # --use_migraphx \

    cp ${THIS_DIR}/${build_dir}/${config}/*.nupkg /home/guangyunhan/rocm-dotnet/

fi
