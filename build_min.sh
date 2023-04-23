#!/bin/bash

usage() { echo "Usage $0 [--no-config]" 1>&2; exit 1; }

config_cmake=true

THIS_DIR=$(dirname $(realpath $0))

set -ex


build_dir="build_min${build_dir_suffix}"
# config="RelWithDebInfo"
config="Release"

if $config_cmake; then
    rm -f  ${THIS_DIR}/${build_dir}/${config}/*.so
    # rm -fr ${THIS_DIR}/${build_dir}/${config}/build/lib

    ${THIS_DIR}/build.sh \
        --build_dir ${THIS_DIR}/${build_dir} \
        --config ${config} \
        --minimal_build \
        --cmake_generator Ninja \
        --cmake_extra_defines \
            CMAKE_EXPORT_COMPILE_COMMANDS=ON \
        --use_cache \
        --build_wheel \
        --skip_submodule_sync --skip_tests

    ${THIS_DIR}/create-fake-dist-info.sh ${THIS_DIR}/${build_dir}/${config}/build/lib/
else
    printf "\n\tSkipping config\n\n"
    cmake --build ${THIS_DIR}/${build_dir}/${config}
fi
