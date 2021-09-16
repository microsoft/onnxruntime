#!/bin/bash

# Note: This script is intended to be called from a macOS pipeline to build the host protoc
# See tools/ci_build/github/azure-pipelines/mac-ios-ci-pipeline.yml
# The host_protoc can be found as $PROTOC_INSTALL_PATH/bin/protoc

set -e

if [ $# -ne 3 ]
then
    echo "Usage: ${0} <repo_root_path> <host_protoc_build_path> <host_protoc_install_path>"
    exit 1
fi

set -x

ORT_REPO_ROOT=$1
PROTOC_BUILD_PATH=$2
PROTOC_INSTALL_PATH=$3

pushd .
mkdir -p $PROTOC_BUILD_PATH
cd $PROTOC_BUILD_PATH
cmake $ORT_REPO_ROOT/cmake/external/protobuf/cmake \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_WITH_ZLIB_DEFAULT=OFF \
    -Dprotobuf_BUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$PROTOC_INSTALL_PATH
make -j $(getconf _NPROCESSORS_ONLN)
make install
popd
