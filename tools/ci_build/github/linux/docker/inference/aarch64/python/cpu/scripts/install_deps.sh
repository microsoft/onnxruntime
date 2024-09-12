#!/bin/bash
set -e -x
pushd .
PYTHON_EXES=("/opt/python/cp38-cp38/bin/python3.8" "/opt/python/cp39-cp39/bin/python3.9" "/opt/python/cp310-cp310/bin/python3.10" "/opt/python/cp311-cp311/bin/python3.11" "/opt/python/cp312-cp312/bin/python3.12")
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
    ccache_url="https://github.com/ccache/ccache/archive/refs/tags/v4.8.tar.gz"
    pushd .
    curl -sSL --retry 5 --retry-delay 10 --create-dirs --fail -L -o ccache_src.tar.gz $ccache_url
    mkdir ccache_main
    cd ccache_main
    tar -zxf ../ccache_src.tar.gz --strip=1

    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local _DCMAKE_BUILD_TYPE=Release ..
    make
    make install
    which ccache
    popd
    rm -f ccache_src.tar.gz
    rm -rf ccache_src
fi
