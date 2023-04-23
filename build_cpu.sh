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

build_dir="build_cpu"
config="RelWithDebInfo"


if $config_cmake; then
    rm -f  ${THIS_DIR}/${build_dir}/${config}/*.so
    # rm -fr ${THIS_DIR}/${build_dir}/${config}/build/lib

    ${THIS_DIR}/build.sh \
        --build_dir ${THIS_DIR}/${build_dir} \
        --config ${config} \
        --cmake_generator Ninja \
        --cmake_extra_defines \
            onnxruntime_ENABLE_PYTHON=ON \
            onnxruntime_BUILD_SHARED_LIB=ON \
            onnxruntime_ENABLE_ATEN=ON \
        --build_wheel \
        --skip_submodule_sync --skip_tests


    ${THIS_DIR}/create-fake-dist-info.sh ${THIS_DIR}/${build_dir}/${config}/build/lib/
else
    printf "\n\tSkipping config\n\n"
    cmake --build ${THIS_DIR}/${build_dir}/${config}
fi
