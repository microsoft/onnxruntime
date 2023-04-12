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

if ! [ -x "$(command -v ccache)" ]; then
  echo "Installing CCache"
  mkdir -p /tmp/ccache
  CCACHE_URL="https://github.com/ccache/ccache/releases/download/v4.7.4/ccache-4.7.4-linux-x86_64.tar.xz"
  curl -sSL --retry 5 --retry-delay 10 --create-dirs --fail -L -o /tmp/src/ccache-4.7.4-linux-x86_64.tar.xz  ${CCACHE_URL}
  tar --strip 1 -xf /tmp/src/ccache-4.7.4-linux-x86_64.tar.xz -C /tmp/ccache
  cp /tmp/ccache/ccache /usr/bin
fi
