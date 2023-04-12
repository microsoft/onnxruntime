#!/bin/bash
set -e -x
pushd .
PYTHON_EXES=("/opt/python/cp38-cp38/bin/python3.8" "/opt/python/cp39-cp39/bin/python3.9" "/opt/python/cp310-cp310/bin/python3.10" "/opt/python/cp311-cp311/bin/python3.11")
CURRENT_DIR=$(pwd)
if ! [ -x "$(command -v protoc)" ]; then
  $CURRENT_DIR/install_protobuf.sh
fi
popd
export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"

for PYTHON_EXE in "${PYTHON_EXES[@]}"
do
  ${PYTHON_EXE} -m pip install -r requirements.txt
done

# No release binary for ccache aarch64, so we need to build it from source.
if ! [ -x "$(command -v ccache)" ]; then
    git clone --recursive --branch v4.7.4 https://github.com/ccache/ccache
    pushd .
    cd ccache
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local _DCMAKE_BUILD_TYPE=Releae ..
    make
    make install
    which ccache
fi
